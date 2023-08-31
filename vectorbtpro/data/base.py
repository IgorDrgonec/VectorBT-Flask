# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with data sources."""

import warnings
from pathlib import Path
import traceback
import inspect
import string

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_any_array, to_pd_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.data.decorators import attach_symbol_dict_methods
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig, copy_dict
from vectorbtpro.utils.datetime_ import is_tz_aware, to_timezone, try_to_datetime_index
from vectorbtpro.utils.parsing import get_func_arg_names, extend_args
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.template import RepEval, CustomTemplate, substitute_templates
from vectorbtpro.utils.pickling import pdict, RecState
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.decorators import cached_property, class_or_instancemethod
from vectorbtpro.utils.selection import _NoResult, NoResult, NoResultsException

__all__ = [
    "key_dict",
    "feature_dict",
    "symbol_dict",
    "run_func_dict",
    "run_arg_dict",
    "Data",
]

__pdoc__ = {}


class key_dict(pdict):
    """Dict that contains features or symbols as keys."""

    pass


class feature_dict(key_dict):
    """Dict that contains features as keys."""

    pass


class symbol_dict(key_dict):
    """Dict that contains symbols as keys."""

    pass


class run_func_dict(pdict):
    """Dict that contains function names as keys for `Data.run`."""

    pass


class run_arg_dict(pdict):
    """Dict that contains argument names as keys for `Data.run`."""

    pass


BaseDataMixinT = tp.TypeVar("BaseDataMixinT", bound="BaseDataMixin")


class BaseDataMixin:
    """Base mixin class for working with data."""

    @property
    def feature_wrapper(self) -> ArrayWrapper:
        """Column wrapper."""
        raise NotImplementedError

    @property
    def symbol_wrapper(self) -> ArrayWrapper:
        """Symbol wrapper."""
        raise NotImplementedError

    @property
    def features(self) -> tp.List[tp.Feature]:
        """List of features."""
        return self.feature_wrapper.columns.tolist()

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        """List of symbols."""
        return self.symbol_wrapper.columns.tolist()

    @classmethod
    def has_multiple_keys(cls, keys: tp.MaybeKeys) -> bool:
        """Check whether there are one or multiple keys."""
        if checks.is_hashable(keys):
            return False
        elif checks.is_sequence(keys):
            return True
        raise TypeError("Keys must be either a hashable or a sequence of hashable")

    def get_feature_idx(self, feature: tp.Feature, raise_error: bool = False) -> int:
        """Return the index of a feature."""

        def _prepare_feature(x):
            if isinstance(x, tuple):
                return tuple([_prepare_feature(_x) for _x in x])
            if isinstance(x, str):
                return x.lower().strip().replace(" ", "_")
            return x

        feature = _prepare_feature(feature)

        found_indices = []
        for i, c in enumerate(self.features):
            c = _prepare_feature(c)
            if feature == c:
                found_indices.append(i)
        if len(found_indices) == 0:
            if raise_error:
                raise ValueError(f"No features match the feature '{str(feature)}'")
            return -1
        if len(found_indices) == 1:
            return found_indices[0]
        raise ValueError(f"Multiple features match the feature '{str(feature)}'")

    def get_symbol_idx(self, symbol: tp.Symbol, raise_error: bool = False) -> int:
        """Return the index of a symbol."""

        def _prepare_symbol(x):
            if isinstance(x, tuple):
                return tuple([_prepare_symbol(_x) for _x in x])
            if isinstance(x, str):
                return x.lower().strip().replace(" ", "_")
            return x

        symbol = _prepare_symbol(symbol)

        found_indices = []
        for i, c in enumerate(self.symbols):
            c = _prepare_symbol(c)
            if symbol == c:
                found_indices.append(i)
        if len(found_indices) == 0:
            if raise_error:
                raise ValueError(f"No symbols match the symbol '{str(symbol)}'")
            return -1
        if len(found_indices) == 1:
            return found_indices[0]
        raise ValueError(f"Multiple symbols match the symbol '{str(symbol)}'")

    def select_feature_idxs(self: BaseDataMixinT, idxs: tp.MaybeSequence[int], **kwargs) -> BaseDataMixinT:
        """Select one or more features by index.

        Returns a new instance."""
        raise NotImplementedError

    def select_symbol_idxs(self: BaseDataMixinT, idxs: tp.MaybeSequence[int], **kwargs) -> BaseDataMixinT:
        """Select one or more symbols by index.

        Returns a new instance."""
        raise NotImplementedError

    def select_features(self: BaseDataMixinT, features: tp.Union[tp.MaybeFeatures], **kwargs) -> BaseDataMixinT:
        """Select one or more features.

        Returns a new instance."""
        if self.has_multiple_keys(features):
            feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
        else:
            feature_idxs = self.get_feature_idx(features, raise_error=True)
        return self.select_feature_idxs(feature_idxs, **kwargs)

    def select_symbols(self: BaseDataMixinT, symbols: tp.Union[tp.MaybeSymbols], **kwargs) -> BaseDataMixinT:
        """Select one or more symbols.

        Returns a new instance."""
        if self.has_multiple_keys(symbols):
            symbol_idxs = [self.get_symbol_idx(k, raise_error=True) for k in symbols]
        else:
            symbol_idxs = self.get_symbol_idx(symbols, raise_error=True)
        return self.select_symbol_idxs(symbol_idxs, **kwargs)

    def get(
        self,
        features: tp.Optional[tp.MaybeFeatures] = None,
        symbols: tp.Optional[tp.MaybeSymbols] = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Get one or more features of one or more symbols of data."""
        raise NotImplementedError

    def has_feature(self, feature: tp.Feature) -> bool:
        """Whether feature exists."""
        feature_idx = self.get_feature_idx(feature, raise_error=False)
        return feature_idx != -1

    def has_symbol(self, symbol: tp.Symbol) -> bool:
        """Whether symbol exists."""
        symbol_idx = self.get_symbol_idx(symbol, raise_error=False)
        return symbol_idx != -1

    def assert_has_feature(self, feature: tp.Feature) -> None:
        """Assert that feature exists."""
        self.get_feature_idx(feature, raise_error=True)

    def assert_has_symbol(self, symbol: tp.Symbol) -> None:
        """Assert that symbol exists."""
        self.get_symbol_idx(symbol, raise_error=True)

    def get_feature(
        self,
        feature: tp.Union[int, tp.Feature],
        raise_error: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get feature that match a feature index or label."""
        if checks.is_int(feature):
            return self.get(features=self.features[feature])
        feature_idx = self.get_feature_idx(feature, raise_error=raise_error)
        if feature_idx == -1:
            return None
        return self.get(features=self.features[feature_idx])

    def get_symbol(
        self,
        symbol: tp.Union[int, tp.Symbol],
        raise_error: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get symbol that match a symbol index or label."""
        if checks.is_int(symbol):
            return self.get(symbol=self.symbols[symbol])
        symbol_idx = self.get_symbol_idx(symbol, raise_error=raise_error)
        if symbol_idx == -1:
            return None
        return self.get(symbol=self.symbols[symbol_idx])


OHLCDataMixinT = tp.TypeVar("OHLCDataMixinT", bound="OHLCDataMixin")


class OHLCDataMixin(BaseDataMixin):
    """Mixin class for working with OHLC data."""

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open."""
        return self.get_feature("Open")

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High."""
        return self.get_feature("High")

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low."""
        return self.get_feature("Low")

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close."""
        return self.get_feature("Close")

    @property
    def volume(self) -> tp.Optional[tp.SeriesFrame]:
        """Volume."""
        return self.get_feature("Volume")

    @property
    def trade_count(self) -> tp.Optional[tp.SeriesFrame]:
        """Trade count."""
        return self.get_feature("Trade count")

    @property
    def vwap(self) -> tp.Optional[tp.SeriesFrame]:
        """VWAP."""
        return self.get_feature("VWAP")

    @property
    def hlc3(self) -> tp.SeriesFrame:
        """HLC/3."""
        high = self.get_feature("High", raise_error=True)
        low = self.get_feature("Low", raise_error=True)
        close = self.get_feature("Close", raise_error=True)
        return (high + low + close) / 3

    @property
    def ohlc4(self) -> tp.SeriesFrame:
        """OHLC/4."""
        open = self.get_feature("Open", raise_error=True)
        high = self.get_feature("High", raise_error=True)
        low = self.get_feature("Low", raise_error=True)
        close = self.get_feature("Close", raise_error=True)
        return (open + high + low + close) / 4

    @property
    def has_ohlc(self) -> bool:
        """Whether the instance has all the OHLC features."""
        return (
            self.has_feature("Open")
            and self.has_feature("High")
            and self.has_feature("Low")
            and self.has_feature("Close")
        )

    @property
    def has_ohlcv(self) -> bool:
        """Whether the instance has all the OHLCV features."""
        return self.has_ohlc and self.has_feature("Volume")

    @property
    def ohlc(self: OHLCDataMixinT) -> OHLCDataMixinT:
        """Return a `OHLCDataMixin` instance with the OHLC features only."""
        open_idx = self.get_feature_idx("Open", raise_error=True)
        high_idx = self.get_feature_idx("High", raise_error=True)
        low_idx = self.get_feature_idx("Low", raise_error=True)
        close_idx = self.get_feature_idx("Close", raise_error=True)
        return self.select_feature_idxs([open_idx, high_idx, low_idx, close_idx])

    @property
    def ohlcv(self: OHLCDataMixinT) -> OHLCDataMixinT:
        """Return a `OHLCDataMixin` instance with the OHLCV features only."""
        open_idx = self.get_feature_idx("Open", raise_error=True)
        high_idx = self.get_feature_idx("High", raise_error=True)
        low_idx = self.get_feature_idx("Low", raise_error=True)
        close_idx = self.get_feature_idx("Close", raise_error=True)
        volume_idx = self.get_feature_idx("Volume", raise_error=True)
        return self.select_feature_idxs([open_idx, high_idx, low_idx, close_idx, volume_idx])

    def get_returns_acc(self, **kwargs) -> ReturnsAccessor:
        """Return accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`."""
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            **kwargs,
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """`OHLCDataMixin.get_returns_acc` with default arguments."""
        return self.get_returns_acc()

    def get_returns(self, **kwargs) -> tp.SeriesFrame:
        """Returns."""
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
            **kwargs,
        )

    @property
    def returns(self) -> tp.SeriesFrame:
        """`OHLCDataMixin.get_returns` with default arguments."""
        return self.get_returns()

    def get_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Log returns."""
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
            log_returns=True,
            **kwargs,
        )

    @property
    def log_returns(self) -> tp.SeriesFrame:
        """`OHLCDataMixin.get_log_returns` with default arguments."""
        return self.get_log_returns()

    def get_daily_returns(self, **kwargs) -> tp.SeriesFrame:
        """Daily returns."""
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            **kwargs,
        ).daily()

    @property
    def daily_returns(self) -> tp.SeriesFrame:
        """`OHLCDataMixin.get_daily_returns` with default arguments."""
        return self.get_daily_returns()

    def get_daily_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Daily log returns."""
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            log_returns=True,
            **kwargs,
        ).daily()

    @property
    def daily_log_returns(self) -> tp.SeriesFrame:
        """`OHLCDataMixin.get_daily_log_returns` with default arguments."""
        return self.get_daily_log_returns()

    def get_drawdowns(self, **kwargs) -> Drawdowns:
        """Generate drawdown records.

        See `vectorbtpro.generic.drawdowns.Drawdowns`."""
        return Drawdowns.from_price(
            open=self.get_feature("Open", raise_error=True),
            high=self.get_feature("High", raise_error=True),
            low=self.get_feature("Low", raise_error=True),
            close=self.get_feature("Close", raise_error=True),
            **kwargs,
        )

    @property
    def drawdowns(self) -> Drawdowns:
        """`OHLCDataMixin.get_drawdowns` with default arguments."""
        return self.get_drawdowns()


DataT = tp.TypeVar("DataT", bound="Data")


class MetaFeatures(type):
    """Meta class that exposes a read-only class property `MetaFeatures.feature_config`."""

    @property
    def feature_config(cls) -> Config:
        """Column config."""
        return cls._feature_config


class DataWithFeatures(metaclass=MetaFeatures):
    """Class exposes a read-only class property `DataWithFeatures.field_config`."""

    @property
    def feature_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${feature_config}
        ```
        """
        return self._feature_config


class MetaData(type(Analyzable), type(DataWithFeatures)):
    pass


@attach_symbol_dict_methods
class Data(Analyzable, DataWithFeatures, OHLCDataMixin, metaclass=MetaData):
    """Class that downloads, updates, and manages data coming from a data source."""

    _setting_keys: tp.SettingsKeys = dict(base="data")

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_feature_config"}

    _feature_config: tp.ClassVar[Config] = HybridConfig()

    _key_dict_attrs = [
        "fetch_kwargs",
        "returned_kwargs",
        "last_index",
        "delisted",
        "classes",
    ]
    """Attributes that subclass either `feature_dict` or `symbol_dict`."""

    _data_dict_type_attrs = [
        "classes",
    ]
    """Attributes that subclass the data dict type."""

    @property
    def feature_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${feature_config}
        ```

        Returns `${cls_name}._feature_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change fields, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._feature_config`.
        """
        return self._feature_config

    def use_feature_config_of(self, cls: tp.Type[DataT]) -> None:
        """Copy feature config from another `Data` class."""
        self._feature_config = cls.feature_config.copy()

    @classmethod
    def modify_state(cls, rec_state: RecState) -> RecState:
        # Ensure backward compatibility
        if "_column_config" in rec_state.attr_dct and "_feature_config" not in rec_state.attr_dct:
            new_attr_dct = dict(rec_state.attr_dct)
            new_attr_dct["_feature_config"] = new_attr_dct.pop("_column_config")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=rec_state.init_kwargs,
                attr_dct=new_attr_dct,
            )
        if "single_symbol" in rec_state.init_kwargs and "single_key" not in rec_state.init_kwargs:
            new_init_kwargs = dict(rec_state.init_kwargs)
            new_init_kwargs["single_key"] = new_init_kwargs.pop("single_symbol")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=new_init_kwargs,
                attr_dct=rec_state.attr_dct,
            )
        if "symbol_classes" in rec_state.init_kwargs and "classes" not in rec_state.init_kwargs:
            new_init_kwargs = dict(rec_state.init_kwargs)
            new_init_kwargs["classes"] = new_init_kwargs.pop("symbol_classes")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=new_init_kwargs,
                attr_dct=rec_state.attr_dct,
            )
        return rec_state

    @classmethod
    def fix_data_dict_type(cls, data: dict) -> tp.Union[feature_dict, symbol_dict]:
        """Fix dict type for data."""
        checks.assert_instance_of(data, dict, arg_name="data")
        if not isinstance(data, key_dict):
            data = symbol_dict(data)
        return data

    @classmethod
    def fix_dict_types_in_kwargs(
        cls,
        data_type: tp.Type[tp.Union[feature_dict, symbol_dict]],
        **kwargs: tp.Kwargs,
    ) -> tp.Kwargs:
        """Fix dict types in keyword arguments."""
        for attr in cls._key_dict_attrs:
            if attr in kwargs:
                attr_value = kwargs[attr]
                checks.assert_instance_of(attr_value, dict, arg_name=attr)
                if not isinstance(attr_value, key_dict):
                    attr_value = data_type(attr_value)
                if attr in cls._data_dict_type_attrs:
                    checks.assert_instance_of(attr_value, data_type, arg_name=attr)
                kwargs[attr] = attr_value
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers."""
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

        keys = set()
        for obj in objs:
            keys = keys.union(set(obj.data.keys()))
        data_type = None
        for obj in objs:
            if len(keys.difference(set(obj.data.keys()))) > 0:
                if isinstance(obj.data, feature_dict):
                    raise ValueError("Objects to be merged must have the same features")
                else:
                    raise ValueError("Objects to be merged must have the same symbols")
            if data_type is None:
                data_type = type(obj.data)
            elif not isinstance(obj.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
        if "data" not in kwargs:
            new_data = data_type()
            for k in objs[0].data.keys():
                new_data[k] = kwargs["wrapper"].row_stack_arrs(*[obj.data[k] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        kwargs["data"] = cls.fix_data_dict_type(kwargs["data"])
        for attr in cls._key_dict_attrs:
            if attr not in kwargs:
                attr_data_type = None
                for obj in objs:
                    v = getattr(obj, attr)
                    if attr_data_type is None:
                        attr_data_type = type(v)
                    elif not isinstance(v, attr_data_type):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                kwargs[attr] = getattr(objs[-1], attr)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        kwargs = cls.fix_dict_types_in_kwargs(type(kwargs["data"]), **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers."""
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
                **wrapper_kwargs,
            )

        keys = set()
        for obj in objs:
            keys = keys.union(set(obj.data.keys()))
        data_type = None
        for obj in objs:
            if len(keys.difference(set(obj.data.keys()))) > 0:
                if isinstance(obj.data, feature_dict):
                    raise ValueError("Objects to be merged must have the same features")
                else:
                    raise ValueError("Objects to be merged must have the same symbols")
            if data_type is None:
                data_type = type(obj.data)
            elif not isinstance(obj.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
        if "data" not in kwargs:
            new_data = data_type()
            for k in objs[0].data.keys():
                new_data[k] = kwargs["wrapper"].column_stack_arrs(*[obj.data[k] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        kwargs["data"] = cls.fix_data_dict_type(kwargs["data"])
        for attr in cls._key_dict_attrs:
            if attr not in kwargs:
                attr_data_type = None
                for obj in objs:
                    v = getattr(obj, attr)
                    if attr_data_type is None:
                        attr_data_type = type(v)
                    elif not isinstance(v, attr_data_type):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                if (issubclass(data_type, feature_dict) and issubclass(attr_data_type, symbol_dict)) or (
                    issubclass(data_type, symbol_dict) and issubclass(attr_data_type, feature_dict)
                ):
                    kwargs[attr] = attr_data_type()
                    for obj in objs:
                        kwargs[attr].update(**getattr(obj, attr))

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        kwargs = cls.fix_dict_types_in_kwargs(type(kwargs["data"]), **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "data",
        "single_key",
        "classes",
        "level_name",
        "fetch_kwargs",
        "returned_kwargs",
        "last_index",
        "delisted",
        "tz_localize",
        "tz_convert",
        "missing_index",
        "missing_columns",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        data: tp.Union[feature_dict, symbol_dict],
        single_key: bool = True,
        classes: tp.Union[None, feature_dict, symbol_dict] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        fetch_kwargs: tp.Union[None, feature_dict, symbol_dict] = None,
        returned_kwargs: tp.Union[None, feature_dict, symbol_dict] = None,
        last_index: tp.Union[None, feature_dict, symbol_dict] = None,
        delisted: tp.Union[None, feature_dict, symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            data=data,
            single_key=single_key,
            classes=classes,
            level_name=level_name,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

        data = self.fix_data_dict_type(data)
        for obj in data.values():
            checks.assert_meta_equal(obj, data[list(data.keys())[0]])
        if len(data) > 1:
            single_key = False

        self._data = data
        self._single_key = single_key
        self._level_name = level_name
        self._tz_localize = tz_localize
        self._tz_convert = tz_convert
        self._missing_index = missing_index
        self._missing_columns = missing_columns

        attr_kwargs = dict()
        for attr in self._key_dict_attrs:
            attr_value = locals()[attr]
            attr_kwargs[attr] = attr_value
        attr_kwargs = self.fix_dict_types_in_kwargs(type(data), **attr_kwargs)
        for k, v in attr_kwargs.items():
            setattr(self, "_" + k, v)

        # Copy writeable attrs
        self._feature_config = type(self)._feature_config.copy()

    def replace(self: DataT, **kwargs) -> DataT:
        """See `vectorbtpro.utils.config.Configured.replace`.

        Replaces the data's index and/or columns if they were changed in the wrapper."""
        if "wrapper" in kwargs and "data" not in kwargs:
            wrapper = kwargs["wrapper"]
            if isinstance(wrapper, dict):
                new_index = wrapper.get("index", self.wrapper.index)
                new_columns = wrapper.get("columns", self.wrapper.columns)
            else:
                new_index = wrapper.index
                new_columns = wrapper.columns
            data = self.config["data"]
            new_data = {}
            index_changed = False
            columns_changed = False
            for k, v in data.items():
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    if not v.index.equals(new_index):
                        v = v.copy(deep=False)
                        v.index = new_index
                        index_changed = True
                    if isinstance(v, pd.DataFrame):
                        if not v.columns.equals(new_columns):
                            v = v.copy(deep=False)
                            v.columns = new_columns
                            columns_changed = True
                new_data[k] = v
            if index_changed or columns_changed:
                kwargs["data"] = self.fix_data_dict_type(new_data)
                if columns_changed:
                    rename = dict(zip(self.keys, new_columns))
                    for attr in self._key_dict_attrs:
                        if attr not in kwargs:
                            attr_value = getattr(self, attr)
                            if (self.feature_oriented and isinstance(attr_value, symbol_dict)) or (
                                self.symbol_oriented and isinstance(attr_value, feature_dict)
                            ):
                                kwargs[attr] = self.rename_in_dict(getattr(self, attr), rename)

        kwargs = self.fix_dict_types_in_kwargs(type(kwargs.get("data", self.data)), **kwargs)
        return Analyzable.replace(self, **kwargs)

    def indexing_func(self: DataT, *args, **kwargs) -> DataT:
        """Perform indexing on `Data`."""
        wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_wrapper = wrapper_meta["new_wrapper"]
        new_data = self.dict_type()
        for k, v in self._data.items():
            if wrapper_meta["rows_changed"]:
                v = v.iloc[wrapper_meta["row_idxs"]]
            if wrapper_meta["columns_changed"]:
                v = v.iloc[:, wrapper_meta["col_idxs"]]
            new_data[k] = v
        attr_dicts = dict()
        attr_dicts["last_index"] = type(self.last_index)()
        for k in self.last_index:
            attr_dicts["last_index"][k] = min([self.last_index[k], new_wrapper.index[-1]])
        if wrapper_meta["columns_changed"]:
            new_symbols = new_wrapper.columns
            for attr in self._key_dict_attrs:
                attr_value = getattr(self, attr)
                if (self.feature_oriented and isinstance(attr_value, symbol_dict)) or (
                    self.symbol_oriented and isinstance(attr_value, feature_dict)
                ):
                    if attr in attr_dicts:
                        attr_dicts[attr] = self.select_from_dict(attr_dicts[attr], new_symbols)
                    else:
                        attr_dicts[attr] = self.select_from_dict(attr_value, new_symbols)
        return self.replace(wrapper=new_wrapper, data=new_data, **attr_dicts)

    @property
    def data(self) -> tp.Union[feature_dict, symbol_dict]:
        """Data dictionary.

        Has the type `feature_dict` for feature-oriented data or `symbol_dict` for symbol-oriented data."""
        return self._data

    @property
    def dict_type(self) -> tp.Type[tp.Union[feature_dict, symbol_dict]]:
        """Return the dict type."""
        return type(self.data)

    @property
    def column_type(self) -> tp.Type[tp.Union[feature_dict, symbol_dict]]:
        """Return the column type."""
        if isinstance(self.data, feature_dict):
            return symbol_dict
        return feature_dict

    @property
    def feature_oriented(self) -> bool:
        """Whether data has features as keys."""
        return issubclass(self.dict_type, feature_dict)

    @property
    def symbol_oriented(self) -> bool:
        """Whether data has symbols as keys."""
        return issubclass(self.dict_type, symbol_dict)

    def get_keys(self, dict_type: tp.Type[tp.Union[feature_dict, symbol_dict]]) -> tp.List[tp.Key]:
        """Get keys depending on the provided dict type."""
        checks.assert_subclass_of(dict_type, (feature_dict, symbol_dict), arg_name="dict_type")
        if issubclass(dict_type, feature_dict):
            return self.features
        return self.symbols

    @property
    def keys(self) -> tp.List[tp.Union[tp.Feature, tp.Symbol]]:
        """Keys in data.

        Features if `feature_dict` and symbols if `symbol_dict`."""
        return list(self.data.keys())

    @property
    def single_key(self) -> bool:
        """Whether there is only one key in `Data.data`."""
        return self._single_key

    @property
    def single_feature(self) -> bool:
        """Whether there is only one feature in `Data.data`."""
        if self.feature_oriented:
            return self.single_key
        return self.wrapper.ndim == 1

    @property
    def single_symbol(self) -> bool:
        """Whether there is only one symbol in `Data.data`."""
        if self.symbol_oriented:
            return self.single_key
        return self.wrapper.ndim == 1

    @property
    def classes(self) -> tp.Union[feature_dict, symbol_dict]:
        """Key classes."""
        return self._classes

    @property
    def feature_classes(self) -> tp.Optional[feature_dict]:
        """Feature classes."""
        if self.feature_oriented:
            return self.classes
        return None

    @property
    def symbol_classes(self) -> tp.Optional[symbol_dict]:
        """Symbol classes."""
        if self.symbol_oriented:
            return self.classes
        return None

    @property
    def level_name(self) -> tp.Optional[tp.MaybeIterable[tp.Hashable]]:
        """Level name(s) for keys.

        Keys are symbols or features depending on the data dict type.

        Must be a sequence if keys are tuples, otherwise a hashable.
        If False, no level names will be used."""
        level_name = self._level_name
        first_key = self.keys[0]
        if isinstance(level_name, bool):
            if level_name:
                level_name = None
            else:
                return None
        if self.feature_oriented:
            key_prefix = "feature"
        else:
            key_prefix = "symbol"
        if isinstance(first_key, tuple):
            if level_name is None:
                level_name = ["%s_%d" % (key_prefix, i) for i in range(len(first_key))]
            if not checks.is_iterable(level_name) or isinstance(level_name, str):
                raise TypeError("Level name should be list-like for a MultiIndex")
            return tuple(level_name)
        if level_name is None:
            level_name = key_prefix
        return level_name

    @property
    def fetch_kwargs(self) -> tp.Union[feature_dict, symbol_dict]:
        """Keyword arguments of type `symbol_dict` initially passed to `Data.fetch_symbol`."""
        return self._fetch_kwargs

    @property
    def returned_kwargs(self) -> tp.Union[feature_dict, symbol_dict]:
        """Keyword arguments of type `symbol_dict` returned by `Data.fetch_symbol`."""
        return self._returned_kwargs

    @property
    def last_index(self) -> tp.Union[feature_dict, symbol_dict]:
        """Last fetched index per symbol of type `symbol_dict`."""
        return self._last_index

    @property
    def delisted(self) -> tp.Union[feature_dict, symbol_dict]:
        """Delisted flag per symbol of type `symbol_dict`."""
        return self._delisted

    @property
    def tz_localize(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """Timezone to localize a datetime-naive index to, which is initially passed to `Data.pull`."""
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """Timezone to convert a datetime-aware to, which is initially passed to `Data.pull`."""
        return self._tz_convert

    @property
    def missing_index(self) -> tp.Optional[str]:
        """Argument `missing` passed to `Data.align_index`."""
        return self._missing_index

    @property
    def missing_columns(self) -> tp.Optional[str]:
        """Argument `missing` passed to `Data.align_columns`."""
        return self._missing_columns

    # ############# Getting ############# #

    def get_key_wrapper(
        self,
        keys: tp.Optional[tp.MaybeKeys] = None,
        stack_classes: bool = True,
        index_stack_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ArrayWrapper:
        """Get wrapper with keys as columns."""
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        if keys is None:
            keys = self.keys
            ndim = 1 if self.single_key else 2
        else:
            if self.has_multiple_keys(keys):
                ndim = 2
            else:
                keys = [keys]
                ndim = 1
            for key in keys:
                if self.feature_oriented:
                    self.assert_has_feature(key)
                else:
                    self.assert_has_symbol(key)
        if isinstance(self.level_name, tuple):
            key_columns = pd.MultiIndex.from_tuples(keys, names=self.level_name)
        else:
            key_columns = pd.Index(keys, name=self.level_name)
        wrapper = self.wrapper.replace(
            columns=key_columns,
            ndim=ndim,
            grouper=None,
            **kwargs,
        )

        if stack_classes:
            classes = []
            all_have_classes = True
            for key in wrapper.columns:
                if key in self.classes:
                    key_classes = self.classes[key]
                    if len(key_classes) > 0:
                        classes.append(key_classes)
                    else:
                        all_have_classes = False
                else:
                    all_have_classes = False
            if len(classes) > 0 and not all_have_classes:
                if self.feature_oriented:
                    raise ValueError("Some features have classes while others not")
                else:
                    raise ValueError("Some symbols have classes while others not")
            if len(classes) > 0:
                classes_frame = pd.DataFrame(classes)
                if len(classes_frame.columns) == 1:
                    classes_columns = pd.Index(classes_frame.iloc[:, 0])
                else:
                    classes_columns = pd.MultiIndex.from_frame(classes_frame)
                key_columns = stack_indexes((classes_columns, wrapper.columns), **index_stack_kwargs)
                wrapper = wrapper.replace(columns=key_columns)
        return wrapper

    def get_feature_wrapper(self, features: tp.Optional[tp.MaybeFeatures] = None, **kwargs) -> ArrayWrapper:
        """Get wrapper with features as columns."""
        if self.feature_oriented:
            return self.get_key_wrapper(keys=features, **kwargs)
        wrapper = self.wrapper
        if features is not None:
            wrapper = wrapper[features]
        return wrapper

    @cached_property
    def feature_wrapper(self) -> ArrayWrapper:
        return self.get_feature_wrapper()

    def get_symbol_wrapper(self, symbols: tp.Optional[tp.MaybeSymbols] = None, **kwargs) -> ArrayWrapper:
        """Get wrapper with symbols as columns."""
        if self.symbol_oriented:
            return self.get_key_wrapper(keys=symbols, **kwargs)
        wrapper = self.wrapper
        if symbols is not None:
            wrapper = wrapper[symbols]
        return wrapper

    @cached_property
    def symbol_wrapper(self) -> ArrayWrapper:
        return self.get_symbol_wrapper()

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object was two-dimensional.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.shape_2d

    @property
    def columns(self) -> tp.Index:
        """Columns.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.columns

    @property
    def index(self) -> tp.Index:
        """Index.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.index

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """Frequency.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.freq

    @property
    def features(self) -> tp.List[tp.Feature]:
        if self.feature_oriented:
            return self.keys
        return self.wrapper.columns.tolist()

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        if self.feature_oriented:
            return self.wrapper.columns.tolist()
        return self.keys

    def select_feature_idxs(self: DataT, idxs: tp.MaybeSequence[int], **kwargs) -> DataT:
        if self.feature_oriented:
            if checks.is_int(idxs):
                return self.select(self.keys[idxs], **kwargs)
            return self.select([self.keys[i] for i in idxs], **kwargs)

        return self.iloc[:, idxs]

    def select_symbol_idxs(self: DataT, idxs: tp.MaybeSequence[int], **kwargs) -> DataT:
        if self.feature_oriented:
            return self.iloc[:, idxs]

        if checks.is_int(idxs):
            return self.select(self.keys[idxs], **kwargs)
        return self.select([self.keys[i] for i in idxs], **kwargs)

    def concat(
        self,
        keys: tp.Optional[tp.Symbols] = None,
        stack_classes: bool = True,
        index_stack_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[feature_dict, symbol_dict]:
        """Concatenate keys along columns."""
        key_wrapper = self.get_key_wrapper(
            keys=keys,
            stack_classes=stack_classes,
            index_stack_kwargs=index_stack_kwargs,
            **kwargs,
        )
        if keys is None:
            keys = self.keys

        new_data = self.column_type()
        first_data = self.data[keys[0]]
        if key_wrapper.ndim == 1:
            if isinstance(first_data, pd.Series):
                new_data[first_data.name] = key_wrapper.wrap(first_data.values, zero_to_none=False)
            else:
                for c in first_data.columns:
                    new_data[c] = key_wrapper.wrap(first_data[c].values, zero_to_none=False)
        else:
            if isinstance(first_data, pd.Series):
                columns = pd.Index([first_data.name])
            else:
                columns = first_data.columns
            for c in columns:
                col_data = []
                for k in keys:
                    if isinstance(self.data[k], pd.Series):
                        col_data.append(self.data[k].values)
                    else:
                        col_data.append(self.data[k][c].values)
                new_data[c] = key_wrapper.wrap(np.column_stack(col_data), zero_to_none=False)

        return new_data

    def get(
        self,
        features: tp.Optional[tp.MaybeFeatures] = None,
        symbols: tp.Optional[tp.MaybeSymbols] = None,
        squeeze_features: bool = False,
        squeeze_symbols: bool = False,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Get one or more features of one or more symbols of data."""
        if features is not None:
            if squeeze_features and self.has_multiple_keys(features) and len(features) == 1:
                features = features[0]
            if self.has_multiple_keys(features):
                feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
                features = [self.features[i] for i in feature_idxs]
                single_feature = False
            else:
                feature_idxs = self.get_feature_idx(features, raise_error=True)
                features = self.features[feature_idxs]
                single_feature = True
        else:
            single_feature = self.single_feature
            if squeeze_features and not single_feature and len(self.features) == 1:
                single_feature = True
            if single_feature:
                feature_idxs = 0
                features = self.features[feature_idxs]
            else:
                feature_idxs = list(range(len(self.features)))
                features = self.features
        if symbols is not None:
            if squeeze_symbols and self.has_multiple_keys(symbols) and len(symbols) == 1:
                symbols = symbols[0]
            if self.has_multiple_keys(symbols):
                symbol_idxs = [self.get_symbol_idx(k, raise_error=True) for k in symbols]
                symbols = [self.symbols[i] for i in symbol_idxs]
                single_symbol = False
            else:
                symbol_idxs = self.get_symbol_idx(symbols, raise_error=True)
                symbols = self.symbols[symbol_idxs]
                single_symbol = True
        else:
            single_symbol = self.single_symbol
            if squeeze_symbols and not single_symbol and len(self.symbols) == 1:
                single_symbol = True
            if single_symbol:
                symbol_idxs = 0
                symbols = self.symbols[symbol_idxs]
            else:
                symbol_idxs = list(range(len(self.symbols)))
                symbols = self.symbols

        if self.feature_oriented:
            if single_feature:
                if self.single_symbol:
                    return self.data[self.features[feature_idxs]]
                return self.data[self.features[feature_idxs]].iloc[:, symbol_idxs]
            concat_data = self.concat(keys=features, **kwargs)
            if single_symbol:
                return list(concat_data.values())[symbol_idxs]
            return tuple([list(concat_data.values())[i] for i in symbol_idxs])
        else:
            if single_symbol:
                if self.single_feature:
                    return self.data[self.symbols[symbol_idxs]]
                return self.data[self.symbols[symbol_idxs]].iloc[:, feature_idxs]
            concat_data = self.concat(keys=symbols, **kwargs)
            if single_feature:
                return list(concat_data.values())[feature_idxs]
            return tuple([list(concat_data.values())[i] for i in feature_idxs])

    # ############# Pre- and post-processing ############# #

    @classmethod
    def prepare_tzaware_index(
        cls,
        obj: tp.SeriesFrame,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
    ) -> tp.SeriesFrame:
        """Prepare a timezone-aware index of a pandas object.

        If the index is tz-naive, convert to a timezone using `tz_localize`.
        Convert the index from one timezone to another using `tz_convert`.
        See `vectorbtpro.utils.datetime_.to_timezone`.

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if tz_localize is None:
            tz_localize = data_cfg["tz_localize"]
        if isinstance(tz_localize, bool):
            if tz_localize:
                raise ValueError("tz_localize cannot be True")
            else:
                tz_localize = None
        if tz_convert is None:
            tz_convert = data_cfg["tz_convert"]
        if isinstance(tz_convert, bool):
            if tz_convert:
                raise ValueError("tz_convert cannot be True")
            else:
                tz_convert = None

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
        data: dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
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

        new_data = {symbol: obj.reindex(index=index) for symbol, obj in data.items()}
        return type(data)(new_data)

    @classmethod
    def align_columns(
        cls,
        data: dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
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
            if isinstance(obj, pd.Series):
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

        new_data = {}
        for symbol, obj in data.items():
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            obj = obj.reindex(columns=columns)
            if not multiple_columns:
                obj = obj[columns[0]]
                if name_is_none:
                    obj = obj.rename(None)
            new_data[symbol] = obj
        return type(data)(new_data)

    def switch_class(
        self,
        new_cls: tp.Type[DataT],
        clear_fetch_kwargs: bool = False,
        clear_returned_kwargs: bool = False,
        **kwargs,
    ) -> DataT:
        """Switch the class of the data instance."""
        if clear_fetch_kwargs:
            new_fetch_kwargs = type(self.fetch_kwargs)({k: {} for k in self.symbols})
        else:
            new_fetch_kwargs = copy_dict(self.fetch_kwargs)
        if clear_returned_kwargs:
            new_returned_kwargs = type(self.returned_kwargs)({k: {} for k in self.symbols})
        else:
            new_returned_kwargs = copy_dict(self.returned_kwargs)
        return self.replace(
            cls_=new_cls,
            fetch_kwargs=new_fetch_kwargs,
            returned_kwargs=new_returned_kwargs,
            **kwargs,
        )

    @classmethod
    def invert_data(cls, dct: tp.Dict[tp.Key, tp.SeriesFrame]) -> tp.Dict[tp.Key, tp.SeriesFrame]:
        """Invert data by swapping keys and columns."""
        if len(dct) == 0:
            return dct
        new_dct = dict()
        for k, v in dct.items():
            if isinstance(v, pd.Series):
                if v.name not in new_dct:
                    new_dct[v.name] = []
                new_dct[v.name].append(v.rename(k))
            else:
                for c in v.columns:
                    if c not in new_dct:
                        new_dct[c] = []
                    new_dct[c].append(v[c].rename(k))
        new_dct2 = {}
        for k, v in new_dct.items():
            if len(v) == 1:
                new_dct2[k] = v[0]
            else:
                new_dct2[k] = pd.concat(v, axis=1)

        if isinstance(dct, symbol_dict):
            return feature_dict(new_dct2)
        if isinstance(dct, feature_dict):
            return symbol_dict(new_dct2)
        return new_dct2

    @class_or_instancemethod
    def align_data(
        cls_or_self,
        data: dict,
        last_index: tp.Union[None, feature_dict, symbol_dict] = None,
        delisted: tp.Union[None, feature_dict, symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
        """Align data.

        Removes any index duplicates, prepares the datetime index, and aligns the index and columns."""
        if last_index is None:
            last_index = {}
        if delisted is None:
            delisted = {}
        if tz_localize is None and not isinstance(cls_or_self, type):
            tz_localize = cls_or_self.tz_localize
        if tz_convert is None and not isinstance(cls_or_self, type):
            tz_convert = cls_or_self.tz_convert
        if missing_index is None and not isinstance(cls_or_self, type):
            missing_index = cls_or_self.missing_index
        if missing_columns is None and not isinstance(cls_or_self, type):
            missing_columns = cls_or_self.missing_columns

        data = type(data)(data)
        for k, obj in data.items():
            obj = to_pd_array(obj)
            obj = obj[~obj.index.duplicated(keep="last")]
            obj = cls_or_self.prepare_tzaware_index(obj, tz_localize=tz_localize, tz_convert=tz_convert)
            data[k] = obj
            if (isinstance(data, symbol_dict) and isinstance(last_index, symbol_dict)) or (
                isinstance(data, feature_dict) and isinstance(last_index, feature_dict)
            ):
                if k not in last_index:
                    last_index[k] = obj.index[-1]
            if (isinstance(data, symbol_dict) and isinstance(delisted, symbol_dict)) or (
                isinstance(data, feature_dict) and isinstance(delisted, feature_dict)
            ):
                if k not in delisted:
                    delisted[k] = False

        data = cls_or_self.align_index(data, missing=missing_index, silence_warnings=silence_warnings)
        data = cls_or_self.align_columns(data, missing=missing_columns, silence_warnings=silence_warnings)

        first_data = data[list(data.keys())[0]]
        if isinstance(first_data, pd.Series):
            columns = [first_data.name]
        else:
            columns = first_data.columns
        for k in columns:
            if (isinstance(data, symbol_dict) and isinstance(last_index, feature_dict)) or (
                isinstance(data, feature_dict) and isinstance(last_index, symbol_dict)
            ):
                if k not in last_index:
                    last_index[k] = first_data.index[-1]
            if (isinstance(data, symbol_dict) and isinstance(delisted, feature_dict)) or (
                isinstance(data, feature_dict) and isinstance(delisted, symbol_dict)
            ):
                if k not in delisted:
                    delisted[k] = False
        for obj in data.values():
            if isinstance(obj.index, pd.DatetimeIndex):
                obj.index.freq = obj.index.inferred_freq

        return data

    @classmethod
    def from_data(
        cls: tp.Type[DataT],
        data: tp.Union[dict, tp.SeriesFrame],
        columns_are_symbols: bool = False,
        invert_data: bool = False,
        single_key: bool = True,
        classes: tp.Optional[dict] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        fetch_kwargs: tp.Optional[dict] = None,
        returned_kwargs: tp.Optional[dict] = None,
        last_index: tp.Optional[dict] = None,
        delisted: tp.Optional[dict] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance from data.

        Args:
            data (dict): Dictionary of array-like objects keyed by symbol.
            columns_are_symbols (bool): Whether columns in each DataFrame are symbols.
            invert_data (bool): Whether to invert the data dictionary with `Data.invert_data`.
            single_key (bool): See `Data.single_key`.
            classes (feature_dict or symbol_dict): See `Data.classes`.
            level_name (bool, hashable or iterable of hashable): See `Data.level_name`.
            tz_localize (timezone_like): See `Data.prepare_tzaware_index`.
            tz_convert (timezone_like): See `Data.prepare_tzaware_index`.
            missing_index (str): See `Data.align_index`.
            missing_columns (str): See `Data.align_columns`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            fetch_kwargs (feature_dict or symbol_dict): Keyword arguments initially passed to `Data.fetch_symbol`.
            returned_kwargs (feature_dict or symbol_dict): Keyword arguments returned by `Data.fetch_symbol`.
            last_index (feature_dict or symbol_dict): Last fetched index per symbol.
            delisted (feature_dict or symbol_dict): Whether symbol has been delisted.
            silence_warnings (bool): Whether to silence all warnings.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbtpro._settings.data`."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if classes is None:
            classes = {}
        if fetch_kwargs is None:
            fetch_kwargs = {}
        if returned_kwargs is None:
            returned_kwargs = {}
        if last_index is None:
            last_index = {}
        if delisted is None:
            delisted = {}

        if columns_are_symbols and isinstance(data, symbol_dict):
            raise TypeError("Data cannot have the type symbol_dict when columns_are_symbols=True")
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if columns_are_symbols:
                data = feature_dict(feature=data)
            else:
                data = symbol_dict(symbol=data)
        checks.assert_instance_of(data, dict, arg_name="data")
        if not isinstance(data, key_dict):
            if columns_are_symbols:
                data = feature_dict(data)
            else:
                data = symbol_dict(data)
        if invert_data:
            data = cls.invert_data(data)
        if len(data) > 1:
            single_key = False
        checks.assert_instance_of(last_index, dict, arg_name="last_index")
        if not isinstance(last_index, key_dict):
            last_index = type(data)(last_index)
        checks.assert_instance_of(delisted, dict, arg_name="delisted")
        if not isinstance(delisted, key_dict):
            delisted = type(data)(delisted)

        data = cls.align_data(
            data,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            silence_warnings=silence_warnings,
        )
        wrapper = ArrayWrapper.from_obj(data[list(data.keys())[0]], **wrapper_kwargs)
        attr_dicts = cls.fix_dict_types_in_kwargs(
            type(data),
            classes=classes,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
        )
        return cls(
            wrapper,
            data,
            single_key=single_key,
            level_name=level_name,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **attr_dicts,
            **kwargs,
        )

    def invert(self: DataT, get_wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Invert data and return a new instance."""
        if get_wrapper_kwargs is None:
            get_wrapper_kwargs = {}
        new_data = self.concat(stack_classes=False)
        if "wrapper" not in kwargs:
            kwargs["wrapper"] = self.get_key_wrapper(**get_wrapper_kwargs)
        if "classes" not in kwargs:
            kwargs["classes"] = self.column_type()
        if "single_key" not in kwargs:
            kwargs["single_key"] = self.wrapper.ndim == 1
        if "level_name" not in kwargs:
            if isinstance(self.wrapper.columns, pd.MultiIndex):
                if self.wrapper.columns.names == [None] * self.wrapper.columns.nlevels:
                    kwargs["level_name"] = False
                else:
                    kwargs["level_name"] = self.wrapper.columns.names
            else:
                if self.wrapper.columns.name is None:
                    kwargs["level_name"] = False
                else:
                    kwargs["level_name"] = self.wrapper.columns.name
        return self.replace(data=new_data, **kwargs)

    def to_feature_oriented(self: DataT, **kwargs) -> DataT:
        """Convert this instance to the feature-oriented format.

        Returns self if the data is already properly formatted."""
        if self.feature_oriented:
            if len(kwargs) > 0:
                return self.replace(**kwargs)
            return self
        return self.invert(**kwargs)

    def to_symbol_oriented(self: DataT, **kwargs) -> DataT:
        """Convert this instance to the symbol-oriented format.

        Returns self if the data is already properly formatted."""
        if self.symbol_oriented:
            if len(kwargs) > 0:
                return self.replace(**kwargs)
            return self
        return self.invert(**kwargs)

    @class_or_instancemethod
    def check_dict_type(
        cls_or_self,
        arg: tp.Any,
        arg_name: str,
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
    ) -> None:
        if isinstance(cls_or_self, type):
            checks.assert_not_none(dict_type, arg_name="dict_type")
        if dict_type is None:
            dict_type = cls_or_self.dict_type
        if issubclass(dict_type, feature_dict) and isinstance(arg, symbol_dict):
            raise TypeError(f"Argument '{arg_name}' must be an instance of feature_dict, not symbol_dict")
        if issubclass(dict_type, symbol_dict) and isinstance(arg, feature_dict):
            raise TypeError(f"Argument '{arg_name}' must be an instance of symbol_dict, not feature_dict")

    @class_or_instancemethod
    def select_key_kwargs(
        cls_or_self,
        key: tp.Key,
        kwargs: tp.DictLike,
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
        check_dict_type: bool = True,
    ) -> dict:
        """Select keyword arguments belonging to a feature or symbol."""
        if isinstance(cls_or_self, type):
            checks.assert_not_none(dict_type, arg_name="dict_type")
        if dict_type is None:
            dict_type = cls_or_self.dict_type
        if kwargs is None:
            return {}
        if check_dict_type:
            cls_or_self.check_dict_type(kwargs, "kwargs", dict_type=dict_type)
        if isinstance(kwargs, (key_dict, dict_type)):
            if key not in kwargs:
                return {}
            kwargs = kwargs[key]
        _kwargs = {}
        for k, v in kwargs.items():
            if check_dict_type:
                cls_or_self.check_dict_type(v, k, dict_type=dict_type)
            if isinstance(v, (key_dict, dict_type)):
                if key in v:
                    _kwargs[k] = v[key]
            else:
                _kwargs[k] = v
        return _kwargs

    @classmethod
    def select_feature_kwargs(cls, feature: tp.Feature, kwargs: tp.DictLike, check_dict_type: bool = True) -> dict:
        """Select keyword arguments belonging to a feature."""
        return cls.select_key_kwargs(feature, kwargs, feature_dict, check_dict_type=check_dict_type)

    @classmethod
    def select_symbol_kwargs(cls, symbol: tp.Symbol, kwargs: tp.DictLike, check_dict_type: bool = True) -> dict:
        """Select keyword arguments belonging to a symbol."""
        return cls.select_key_kwargs(symbol, kwargs, symbol_dict, check_dict_type=check_dict_type)

    @classmethod
    def select_from_dict(cls, dct: dict, keys: tp.Keys, raise_error: bool = False) -> dict:
        """Select keys from a dict."""
        if raise_error:
            return type(dct)({k: dct[k] for k in keys})
        return type(dct)({k: dct[k] for k in keys if k in dct})

    def select(self: DataT, keys: tp.MaybeKeys, **kwargs) -> DataT:
        """Create a new `Data` instance with one or more keys from this instance.

        Applies to features if data has the type `feature_dict` and symbols if `symbol_dict`."""
        if self.has_multiple_keys(keys):
            single_key = False
        else:
            single_key = True
            keys = [keys]
        attr_dicts = dict()
        for attr in self._key_dict_attrs:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, self.dict_type):
                attr_dicts[attr] = self.select_from_dict(attr_value, keys)

        return self.replace(
            data=self.select_from_dict(self.data, keys, raise_error=True),
            single_key=single_key,
            **attr_dicts,
            **kwargs,
        )

    @classmethod
    def rename_in_dict(cls, dct: dict, rename: tp.Dict[tp.Key, tp.Key]) -> dict:
        """Rename keys in a dict."""
        return type(dct)({rename.get(k, k): v for k, v in dct.items()})

    def rename(self: DataT, rename: tp.Dict[tp.Key, tp.Key], **kwargs) -> DataT:
        """Rename keys using `rename` dict that maps old keys to new keys.

        Applies to symbols if data has the type `symbol_dict` and features if `feature_dict`."""
        attr_dicts = dict()
        for attr in self._key_dict_attrs:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, self.dict_type):
                attr_dicts[attr] = self.rename_in_dict(attr_value, rename)
        return self.replace(data=self.rename_in_dict(self.data, rename), **attr_dicts, **kwargs)

    @classmethod
    def merge(
        cls: tp.Type[DataT],
        *datas: DataT,
        rename: tp.Optional[tp.Dict[tp.Key, tp.Key]] = None,
        **kwargs,
    ) -> DataT:
        """Merge multiple `Data` instances.

        Can merge both symbols and features. Data is overridden in the order as provided in `datas`."""
        if len(datas) == 1:
            datas = datas[0]
        datas = list(datas)

        data_type = None
        data = {}
        single_key = True
        attr_dicts = dict()
        for instance in datas:
            if data_type is None:
                data_type = type(instance.data)
            elif not isinstance(instance.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
            if not instance.single_key:
                single_key = False
            for k in instance.keys:
                if rename is None:
                    new_k = k
                else:
                    new_k = rename[k]
                if new_k in data:
                    obj1 = instance.data[k]
                    obj2 = data[new_k]
                    both_were_series = True
                    if isinstance(obj1, pd.Series):
                        obj1 = obj1.to_frame()
                    else:
                        both_were_series = False
                    if isinstance(obj2, pd.Series):
                        obj2 = obj2.to_frame()
                    else:
                        both_were_series = False
                    new_obj = obj1.combine_first(obj2)
                    if new_obj.shape[1] == 1 and both_were_series:
                        new_obj = new_obj.iloc[:, 0]
                    data[new_k] = new_obj
                else:
                    data[new_k] = instance.data[k]
                for attr in cls._key_dict_attrs:
                    attr_value = getattr(instance, attr)
                    if (issubclass(data_type, symbol_dict) and isinstance(attr_value, symbol_dict)) or (
                        issubclass(data_type, feature_dict) and isinstance(attr_value, feature_dict)
                    ):
                        if k in attr_value:
                            if attr not in attr_dicts:
                                attr_dicts[attr] = type(attr_value)()
                            elif not isinstance(attr_value, type(attr_dicts[attr])):
                                raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                            attr_dicts[attr][new_k] = attr_value[k]
            for attr in cls._key_dict_attrs:
                attr_value = getattr(instance, attr)
                if (issubclass(data_type, symbol_dict) and isinstance(attr_value, feature_dict)) or (
                    issubclass(data_type, feature_dict) and isinstance(attr_value, symbol_dict)
                ):
                    if attr not in attr_dicts:
                        attr_dicts[attr] = type(attr_value)()
                    elif not isinstance(attr_value, type(attr_dicts[attr])):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                    attr_dicts[attr].update(**attr_value)

        kwargs = cls.resolve_merge_kwargs(
            *[instance.config for instance in datas],
            wrapper=None,
            data=data_type(data),
            single_key=single_key,
            **attr_dicts,
            **kwargs,
        )
        kwargs.pop("wrapper", None)
        return cls.from_data(**kwargs)

    # ############# Fetching ############# #

    @classmethod
    def fetch_feature(
        cls,
        feature: tp.Feature,
        **kwargs,
    ) -> tp.FeatureData:
        """Fetch a feature.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.
        If there are keyword arguments `tz_localize`, `tz_convert`, or `freq` in this dict,
        will pop them and use them to override global settings.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    @classmethod
    def try_fetch_feature(
        cls,
        feature: tp.Feature,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        fetch_kwargs: tp.KwargsLike = None,
    ) -> tp.FeatureData:
        """Try to fetch a feature using `Data.fetch_feature`."""
        if fetch_kwargs is None:
            fetch_kwargs = {}
        try:
            out = cls.fetch_feature(feature, **fetch_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Feature '{str(feature)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Feature '{str(feature)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Fetch a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.
        If there are keyword arguments `tz_localize`, `tz_convert`, or `freq` in this dict,
        will pop them and use them to override global settings.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    @classmethod
    def try_fetch_symbol(
        cls,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        fetch_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Try to fetch a symbol using `Data.fetch_symbol`."""
        if fetch_kwargs is None:
            fetch_kwargs = {}
        try:
            out = cls.fetch_symbol(symbol, **fetch_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Symbol '{str(symbol)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
    ) -> tp.Kwargs:
        """Resolve metadata for keys."""
        data_cfg = cls.get_settings(key_id="base")

        if keys is not None and features is not None:
            raise ValueError("Must provide either keys or features, not both")
        if keys is not None and symbols is not None:
            raise ValueError("Must provide either keys or symbols, not both")
        if features is not None and symbols is not None:
            raise ValueError("Must provide either features or symbols, not both")
        if keys is None:
            if features is not None:
                keys = features
                keys_are_features = True
                dict_type = feature_dict
            else:
                keys = symbols
                keys_are_features = False
                dict_type = symbol_dict
        else:
            if keys_are_features is None:
                keys_are_features = data_cfg["keys_are_features"]
            if keys_are_features:
                dict_type = feature_dict
            else:
                dict_type = symbol_dict
        return dict(
            keys=keys,
            keys_are_features=keys_are_features,
            dict_type=dict_type,
        )

    @classmethod
    def pull(
        cls: tp.Type[DataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        classes: tp.Optional[tp.MaybeSequence[tp.Union[tp.Hashable, dict]]] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        **kwargs,
    ) -> tp.Union[DataT, tp.List[tp.Any]]:
        """Pull data.

        Fetches each feature/symbol with `Data.fetch_feature`/`Data.fetch_symbol` and prepares it with `Data.from_data`.

        Iteration over features/symbols is done using `vectorbtpro.utils.execution.execute`.
        That is, it can be distributed and parallelized when needed.

        Args:
            keys (hashable, sequence of hashable, or dict): One or multiple keys.

                Depending on `keys_are_features` will be set to `features` or `symbols`.
            keys_are_features (bool): Whether `keys` are considered features.
            features (hashable, sequence of hashable, or dict): One or multiple features.

                If provided as a dictionary, will use keys as features and values as keyword arguments.

                !!! note
                    Tuple is considered as a single feature (tuple is a hashable).
            symbols (hashable, sequence of hashable, or dict): One or multiple symbols.

                If provided as a dictionary, will use keys as symbols and values as keyword arguments.

                !!! note
                    Tuple is considered as a single symbol (tuple is a hashable).
            classes (feature_dict or symbol_dict): See `Data.classes`.

                Can be a hashable (single value), a dictionary (class names as keys and
                class values as values), or a sequence of such.

                !!! note
                    Tuple is considered as a single class (tuple is a hashable).
            level_name (bool, hashable or iterable of hashable): See `Data.level_name`.
            tz_localize (any): See `Data.from_data`.
            tz_convert (any): See `Data.from_data`.
            missing_index (str): See `Data.from_data`.
            missing_columns (str): See `Data.from_data`.
            wrapper_kwargs (dict): See `Data.from_data`.
            skip_on_error (bool): Whether to skip the feature/symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.fetch_feature`/`Data.fetch_symbol` if in the signature.
            execute_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.execution.execute`.
            return_raw (bool): Whether to return the raw outputs.
            **kwargs: Passed to `Data.fetch_feature`/`Data.fetch_symbol`.

                If two features/symbols require different keyword arguments, pass
                `key_dict` or `feature_dict`/`symbol_dict` for each argument.

        For defaults, see `vectorbtpro._settings.data`.
        """
        data_cfg = cls.get_settings(key_id="base")

        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        dict_type = keys_meta["dict_type"]

        fetch_kwargs = dict_type()
        if isinstance(keys, dict):
            new_keys = []
            for k, key_fetch_kwargs in keys.items():
                new_keys.append(k)
                fetch_kwargs[k] = key_fetch_kwargs
            keys = new_keys
            single_key = False
        elif cls.has_multiple_keys(keys):
            keys = list(keys)
            if len(set(keys)) < len(keys):
                raise ValueError("Duplicate keys provided")
            single_key = False
        else:
            single_key = True
            keys = [keys]
        if classes is not None:
            cls.check_dict_type(classes, "classes", dict_type=dict_type)
            if not isinstance(classes, key_dict):
                new_classes = {}
                single_class = checks.is_hashable(classes) or isinstance(classes, dict)
                if single_class:
                    for k in keys:
                        if isinstance(classes, dict):
                            new_classes[k] = classes
                        else:
                            if keys_are_features:
                                new_classes[k] = {"feature_class": classes}
                            else:
                                new_classes[k] = {"symbol_class": classes}
                else:
                    for i, k in enumerate(keys):
                        _classes = classes[i]
                        if not isinstance(_classes, dict):
                            if keys_are_features:
                                _classes = {"feature_class": _classes}
                            else:
                                _classes = {"symbol_class": _classes}
                        new_classes[k] = _classes
                classes = new_classes
        wrapper_kwargs = merge_dicts(data_cfg["wrapper_kwargs"], wrapper_kwargs)
        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]
        execute_kwargs = merge_dicts(data_cfg["execute_kwargs"], execute_kwargs)
        if not single_key and "show_progress" not in execute_kwargs:
            execute_kwargs["show_progress"] = True

        funcs_args = []
        if keys_are_features:
            func_arg_names = get_func_arg_names(cls.fetch_feature)
        else:
            func_arg_names = get_func_arg_names(cls.fetch_symbol)
        for k in keys:
            if keys_are_features:
                key_fetch_func = cls.try_fetch_feature
                key_fetch_kwargs = cls.select_feature_kwargs(k, kwargs)
            else:
                key_fetch_func = cls.try_fetch_symbol
                key_fetch_kwargs = cls.select_symbol_kwargs(k, kwargs)
            if "silence_warnings" in func_arg_names:
                key_fetch_kwargs["silence_warnings"] = silence_warnings
            if k in fetch_kwargs:
                key_fetch_kwargs = merge_dicts(key_fetch_kwargs, fetch_kwargs[k])

            funcs_args.append(
                (
                    key_fetch_func,
                    (k,),
                    dict(
                        skip_on_error=skip_on_error,
                        silence_warnings=silence_warnings,
                        fetch_kwargs=key_fetch_kwargs,
                    ),
                )
            )
            fetch_kwargs[k] = key_fetch_kwargs

        outputs = execute(funcs_args, n_calls=len(keys), progress_desc=keys, **execute_kwargs)
        if return_raw:
            return outputs

        data = dict_type()
        returned_kwargs = dict_type()
        for i, out in enumerate(outputs):
            k = keys[i]
            if out is not None:
                if isinstance(out, tuple):
                    _data = out[0]
                    _returned_kwargs = out[1]
                else:
                    _data = out
                    _returned_kwargs = {}
                _data = to_any_array(_data)
                _tz_localize = _returned_kwargs.pop("tz_localize", None)
                if _tz_localize is not None:
                    if tz_localize is None:
                        tz_localize = _tz_localize
                    elif tz_localize != _tz_localize:
                        raise ValueError("Cannot localize using different timezones")
                _tz_convert = _returned_kwargs.pop("tz_convert", None)
                if _tz_convert is not None:
                    if tz_convert is None:
                        tz_convert = _tz_convert
                    elif tz_convert != _tz_convert:
                        tz_convert = "utc"
                _freq = _returned_kwargs.pop("freq", None)
                if wrapper_kwargs.get("freq", None) is None:
                    wrapper_kwargs["freq"] = _freq
                if _data.size == 0:
                    if not silence_warnings:
                        if keys_are_features:
                            warnings.warn(
                                f"Feature '{str(k)}' returned an empty array. Skipping.",
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                f"Symbol '{str(k)}' returned an empty array. Skipping.",
                                stacklevel=2,
                            )
                else:
                    data[k] = _data
                    returned_kwargs[k] = _returned_kwargs

        if len(data) == 0:
            if keys_are_features:
                raise ValueError("No features could be fetched")
            else:
                raise ValueError("No symbols could be fetched")

        return cls.from_data(
            data,
            single_key=single_key,
            classes=classes,
            level_name=level_name,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            wrapper_kwargs=wrapper_kwargs,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            silence_warnings=silence_warnings,
        )

    @classmethod
    def fetch(cls: tp.Type[DataT], *args, **kwargs) -> tp.Union[DataT, tp.List[tp.Any]]:
        """Exists for backward compatibility. Use `Data.pull` instead."""
        return cls.pull(*args, **kwargs)

    @classmethod
    def from_data_str(cls: tp.Type[DataT], data_str: str) -> DataT:
        """Parse a `Data` instance from a string.

        For example: `YFData:BTC-USD` or just `BTC-USD` where the data class is
        `vectorbtpro.data.custom.yf.YFData` by default."""
        from vectorbtpro.data import custom

        if ":" in data_str:
            cls_name, symbol = data_str.split(":")
            cls_name = cls_name.strip()
            symbol = symbol.strip()
            return getattr(custom, cls_name).pull(symbol)
        return custom.YFData.pull(data_str.strip())

    # ############# Updating ############# #

    def update_feature(
        self,
        feature: tp.Feature,
        **kwargs,
    ) -> tp.FeatureData:
        """Update a feature.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def try_update_feature(
        self,
        feature: tp.Feature,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        update_kwargs: tp.KwargsLike = None,
    ) -> tp.FeatureData:
        """Try to update a feature using `Data.update_feature`."""
        if update_kwargs is None:
            update_kwargs = {}
        try:
            out = self.update_feature(feature, **update_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Feature '{str(feature)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Feature '{str(feature)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    def update_symbol(
        self,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Update a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def try_update_symbol(
        self,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        update_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Try to update a symbol using `Data.update_symbol`."""
        if update_kwargs is None:
            update_kwargs = {}
        try:
            out = self.update_symbol(symbol, **update_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Symbol '{str(symbol)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    def update(
        self: DataT,
        *,
        concat: bool = True,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        **kwargs,
    ) -> tp.Union[DataT, tp.List[tp.Any]]:
        """Update data.

        Fetches new data for each feature/symbol using `Data.update_feature`/`Data.update_symbol`.

        Args:
            concat (bool): Whether to concatenate existing and updated/new data.
            skip_on_error (bool): Whether to skip the feature/symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.update_feature`/`Data.update_symbol`
                if accepted by `Data.fetch_feature`/`Data.fetch_symbol`.
            execute_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.execution.execute`.
            return_raw (bool): Whether to return the raw outputs.
            **kwargs: Passed to `Data.update_feature`/`Data.update_symbol`.

                If two features/symbols require different keyword arguments,
                pass `key_dict` or `feature_dict`/`symbol_dict` for each argument.

        !!! note
            Returns a new `Data` instance instead of changing the data in place.
        """
        data_cfg = self.get_settings(key_id="base")

        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]
        execute_kwargs = merge_dicts(data_cfg["execute_kwargs"], execute_kwargs)
        if "show_progress" not in execute_kwargs:
            execute_kwargs["show_progress"] = False
        if self.feature_oriented:
            func_arg_names = get_func_arg_names(self.fetch_feature)
        else:
            func_arg_names = get_func_arg_names(self.fetch_symbol)
        if "show_progress" in func_arg_names and "show_progress" not in kwargs:
            kwargs["show_progress"] = False
        checks.assert_instance_of(self.last_index, self.dict_type, "last_index")
        checks.assert_instance_of(self.delisted, self.dict_type, "delisted")

        funcs_args = []
        key_indices = []
        for i, k in enumerate(self.keys):
            if not self.delisted.get(k, False):
                if self.feature_oriented:
                    key_update_func = self.try_update_feature
                    key_update_kwargs = self.select_feature_kwargs(k, kwargs)
                else:
                    key_update_func = self.try_update_symbol
                    key_update_kwargs = self.select_symbol_kwargs(k, kwargs)
                if "silence_warnings" in func_arg_names:
                    key_update_kwargs["silence_warnings"] = silence_warnings
                funcs_args.append(
                    (
                        key_update_func,
                        (k,),
                        dict(
                            skip_on_error=skip_on_error,
                            silence_warnings=silence_warnings,
                            update_kwargs=key_update_kwargs,
                        ),
                    )
                )
                key_indices.append(i)

        outputs = execute(funcs_args, n_calls=len(self.keys), progress_desc=self.keys, **execute_kwargs)
        if return_raw:
            return outputs

        new_data = {}
        new_last_index = {}
        new_returned_kwargs = {}
        i = 0
        for k, obj in self.data.items():
            if self.delisted.get(k, False):
                out = None
            else:
                out = outputs[i]
                i += 1
            skip_key = False
            if out is not None:
                if isinstance(out, tuple):
                    new_obj = out[0]
                    new_returned_kwargs[k] = out[1]
                else:
                    new_obj = out
                    new_returned_kwargs[k] = {}
                new_obj = to_any_array(new_obj)
                if new_obj.size == 0:
                    if not silence_warnings:
                        if self.feature_oriented:
                            warnings.warn(
                                f"Feature '{str(k)}' returned an empty array. Skipping.",
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                f"Symbol '{str(k)}' returned an empty array. Skipping.",
                                stacklevel=2,
                            )
                    skip_key = True
                else:
                    if not isinstance(new_obj, (pd.Series, pd.DataFrame)):
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
                    new_data[k] = new_obj
                    if len(new_obj.index) > 0:
                        new_last_index[k] = new_obj.index[-1]
                    else:
                        new_last_index[k] = self.last_index[k]
            else:
                skip_key = True
            if skip_key:
                new_data[k] = obj.iloc[0:0]
                new_last_index[k] = self.last_index[k]

        # Get the last index in the old data from where the new data should begin
        from_index = None
        for k, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                index = new_obj.index[0]
            else:
                continue
            if from_index is None or index < from_index:
                from_index = index
        if from_index is None:
            if not silence_warnings:
                if self.feature_oriented:
                    warnings.warn(
                        f"None of the features were updated",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"None of the symbols were updated",
                        stacklevel=2,
                    )
            return self.copy()

        # Concatenate the updated old data and the new data
        for k, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                to_index = new_obj.index[0]
            else:
                to_index = None
            obj = self.data[k]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                shared_columns = obj.columns.intersection(new_obj.columns)
                obj = obj[shared_columns]
                new_obj = new_obj[shared_columns]
            elif isinstance(new_obj, pd.DataFrame):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            elif isinstance(obj, pd.DataFrame):
                if new_obj.name is not None:
                    obj = obj[new_obj.name]
                else:
                    obj = obj[0]
            obj = obj.loc[from_index:to_index]
            new_obj = pd.concat((obj, new_obj), axis=0)
            new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
            new_data[k] = new_obj

        # Align the index and columns in the new data
        new_data = self.align_index(new_data, missing=self.missing_index, silence_warnings=silence_warnings)
        new_data = self.align_columns(new_data, missing=self.missing_columns, silence_warnings=silence_warnings)

        # Align the columns and data type in the old and new data
        for k, new_obj in new_data.items():
            obj = self.data[k]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                new_obj = new_obj[obj.columns]
            elif isinstance(new_obj, pd.DataFrame):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            if isinstance(obj, pd.DataFrame):
                new_obj = new_obj.astype(obj.dtypes)
            else:
                new_obj = new_obj.astype(obj.dtype)
            new_data[k] = new_obj

        if not concat:
            # Do not concatenate with the old data
            for k, new_obj in new_data.items():
                if isinstance(new_obj.index, pd.DatetimeIndex):
                    new_obj.index.freq = new_obj.index.inferred_freq
            new_index = new_data[self.keys[0]].index
            return self.replace(
                wrapper=self.wrapper.replace(index=new_index),
                data=self.dict_type(new_data),
                returned_kwargs=self.dict_type(new_returned_kwargs),
                last_index=self.dict_type(new_last_index),
            )

        # Append the new data to the old data
        for k, new_obj in new_data.items():
            obj = self.data[k]
            obj = obj.loc[:from_index]
            if obj.index[-1] == from_index:
                obj = obj.iloc[:-1]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if isinstance(new_obj.index, pd.DatetimeIndex):
                new_obj.index.freq = new_obj.index.inferred_freq
            new_data[k] = new_obj

        new_index = new_data[self.keys[0]].index

        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),
            data=self.dict_type(new_data),
            returned_kwargs=self.dict_type(new_returned_kwargs),
            last_index=self.dict_type(new_last_index),
        )

    # ############# Transforming ############# #

    def transform(
        self: DataT,
        transform_func: tp.Callable,
        *args,
        per_feature: bool = False,
        per_symbol: bool = False,
        pass_frame: bool = False,
        **kwargs,
    ) -> DataT:
        """Transform data.

        If one key (i.e., feature or symbol), passes the entire Series/DataFrame. If `per_feature` is True,
        passes the Series/DataFrame of each feature. If `per_symbol` is True, passes the Series/DataFrame
        of each symbol. If both are True, passes each feature and symbol combination as a Series
        if `pass_frame` is False or as a DataFrame with one column if `pass_frame` is True.
        If both are False, concatenates all features and symbols into a single DataFrame
        and calls `transform_func` on it. Then, splits the data by key and builds a new `Data` instance.

        After the transformation, the new data is aligned using `Data.align_data`.

        !!! note
            The returned object must have the same type and dimensionality as the input object.

            Number of columns (i.e., features and symbols) and their names must stay the same.
            To remove columns, use either indexing or `Data.select` (depending on the data orientation).
            To add new columns, use either column stacking or `Data.merge`.

            Index, on the other hand, can be changed freely."""
        if (self.feature_oriented and per_feature) or (self.symbol_oriented and per_symbol):
            new_data = self.dict_type()
            for k in self.keys:
                if (self.feature_oriented and per_symbol) or (self.symbol_oriented and per_feature):
                    if isinstance(self.data[k], pd.Series):
                        new_data[k] = transform_func(self.data[k], *args, **kwargs)
                    else:
                        _new_data = []
                        for i in range(len(self.data[k].columns)):
                            if pass_frame:
                                _new_data.append(transform_func(self.data[k].iloc[:, [i]], *args, **kwargs))
                            else:
                                new_obj = transform_func(self.data[k].iloc[:, i], *args, **kwargs)
                                checks.assert_meta_equal(new_obj, self.data[k].iloc[:, i], axis=1)
                                new_obj = new_obj.to_frame()
                                new_obj.columns = self.data[k].columns[[i]]
                                _new_data.append(new_obj)
                        new_data[k] = pd.concat(_new_data, axis=1)
                else:
                    new_data[k] = transform_func(self.data[k], *args, **kwargs)
                checks.assert_meta_equal(new_data[k], self.data[k], axis=1)
        elif (self.feature_oriented and per_symbol) or (self.symbol_oriented and per_feature):
            first_data = self.data[list(self.data.keys())[0]]
            if isinstance(first_data, pd.Series):
                concat_data = pd.concat(self.data.values(), axis=1)
                new_concat_data = transform_func(concat_data, *args, **kwargs)
                checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
                new_data = self.dict_type()
                for i, k in enumerate(self.keys):
                    new_data[k] = new_concat_data.iloc[:, i]
                    new_data[k].name = first_data.name
            else:
                all_concat_data = []
                for i in range(len(first_data.columns)):
                    concat_data = pd.concat([self.data[k].iloc[:, [i]] for k in self.keys], axis=1)
                    new_concat_data = transform_func(concat_data, *args, **kwargs)
                    checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
                    all_concat_data.append(new_concat_data)
                new_data = self.dict_type()
                for i, k in enumerate(self.keys):
                    new_objs = []
                    for c in range(len(first_data.columns)):
                        new_objs.append(all_concat_data[c].iloc[:, [i]])
                    new_data[k] = pd.concat(new_objs, axis=1)
        else:
            if isinstance(self.level_name, tuple):
                keys = pd.MultiIndex.from_tuples(self.keys, names=self.level_name)
            else:
                keys = pd.Index(self.keys, name=self.level_name)
            concat_data = pd.concat(self.data.values(), axis=1, keys=keys)
            new_concat_data = transform_func(concat_data, *args, **kwargs)
            checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
            new_data = self.dict_type()
            first_data = self.data[list(self.data.keys())[0]]
            for i, k in enumerate(self.keys):
                if isinstance(first_data, pd.Series):
                    new_data[k] = new_concat_data.iloc[:, i]
                    new_data[k].name = first_data.name
                else:
                    start_i = first_data.shape[1] * i
                    stop_i = first_data.shape[1] * (1 + i)
                    new_data[k] = new_concat_data.iloc[:, start_i:stop_i]
                    new_data[k].columns = first_data.columns

        new_data = self.align_data(new_data)
        first_data = new_data[list(new_data.keys())[0]]
        new_wrapper = self.wrapper.replace(index=first_data.index)
        return self.replace(
            wrapper=new_wrapper,
            data=new_data,
        )

    def resample(self: DataT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> DataT:
        """Perform resampling on `Data`.

        Features "open", "high", "low", "close", "volume", "trade count", and "vwap" (case-insensitive)
        are recognized and resampled automatically.

        Looks for `resample_func` of each feature in `Data.feature_config`. The function must
        accept the `Data` instance, object, and resampler."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)

        def _resample_feature(obj, feature, symbol=None):
            resample_func = self.feature_config.get(feature, {}).get("resample_func", None)
            if resample_func is not None:
                if isinstance(resample_func, str):
                    return obj.vbt.resample_apply(wrapper_meta["resampler"], resample_func)
                return resample_func(self, obj, wrapper_meta["resampler"])
            if isinstance(feature, str) and feature.lower() == "open":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.first_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "high":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.max_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "low":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.min_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "close":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.last_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "volume":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.sum_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "trade count":
                return obj.vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.sum_reduce_nb,
                    wrap_kwargs=dict(dtype=int),
                )
            if isinstance(feature, str) and feature.lower() == "vwap":
                volume_obj = None
                for feature2 in self.features:
                    if isinstance(feature2, str) and feature2.lower() == "volume":
                        if self.feature_oriented:
                            volume_obj = self.data[feature2]
                        else:
                            volume_obj = self.data[symbol][feature2]
                if volume_obj is None:
                    raise ValueError("Volume is required to resample VWAP")
                return pd.DataFrame.vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.wmean_range_reduce_meta_nb,
                    to_2d_array(obj),
                    to_2d_array(volume_obj),
                    wrapper=self.wrapper[feature],
                )
            raise ValueError(f"Cannot resample feature '{feature}'. Specify resample_func in feature_config.")

        new_data = self.dict_type()
        if self.feature_oriented:
            for feature in self.features:
                new_data[feature] = _resample_feature(self.data[feature], feature)
        else:
            for symbol, obj in self.data.items():
                _new_obj = []
                for feature in self.features:
                    if self.single_feature:
                        _new_obj.append(_resample_feature(obj, feature, symbol=symbol))
                    else:
                        _new_obj.append(_resample_feature(obj[[feature]], feature, symbol=symbol))
                if self.single_feature:
                    new_obj = _new_obj[0]
                else:
                    new_obj = pd.concat(_new_obj, axis=1)
                new_data[symbol] = new_obj

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    def realign(
        self: DataT,
        rule: tp.Optional[tp.AnyRuleLike] = None,
        *args,
        wrapper_meta: tp.DictLike = None,
        ffill: bool = True,
        **kwargs,
    ) -> DataT:
        """Perform realigning on `Data`.

        Looks for `realign_func` of each feature in `Data.feature_config`. If no function provided,
        resamples feature "open" with `vectorbtpro.generic.accessors.GenericAccessor.resample_opening`
        and other features with `vectorbtpro.generic.accessors.GenericAccessor.resample_closing`."""
        if rule is None:
            rule = self.wrapper.freq
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(rule, *args, **kwargs)

        def _realign_feature(obj, feature, symbol=None):
            realign_func = self.feature_config.get(feature, {}).get("realign_func", None)
            if realign_func is not None:
                if isinstance(realign_func, str):
                    return getattr(obj.vbt, realign_func)(wrapper_meta["resampler"], ffill=ffill)
                return realign_func(self, obj, wrapper_meta["resampler"], ffill=ffill)
            if isinstance(feature, str) and feature.lower() == "open":
                return obj.vbt.resample_opening(wrapper_meta["resampler"], ffill=ffill)
            return obj.vbt.resample_closing(wrapper_meta["resampler"], ffill=ffill)

        new_data = self.dict_type()
        if self.feature_oriented:
            for feature in self.features:
                new_data[feature] = _realign_feature(self.data[feature], feature)
        else:
            for symbol, obj in self.data.items():
                _new_obj = []
                for feature in self.features:
                    if self.single_feature:
                        _new_obj.append(_realign_feature(obj, feature, symbol=symbol))
                    else:
                        _new_obj.append(_realign_feature(obj[[feature]], feature, symbol=symbol))
                if self.single_feature:
                    new_obj = _new_obj[0]
                else:
                    new_obj = pd.concat(_new_obj, axis=1)
                new_data[symbol] = new_obj

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    # ############# Running ############# #

    @classmethod
    def try_run(
        cls,
        data: "Data",
        func_name: str,
        *args,
        raise_errors: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Try to run a function on data."""
        try:
            return data.run(*args, **kwargs)
        except Exception as e:
            if raise_errors:
                raise e
            if not silence_warnings:
                warnings.warn(func_name + ": " + str(e), stacklevel=2)
        return NoResult

    def run(
        self,
        func: tp.MaybeIterable[tp.Union[str, tp.Callable]],
        *args,
        on_features: tp.Optional[tp.MaybeFeatures] = None,
        on_symbols: tp.Optional[tp.MaybeSymbols] = None,
        pass_as_first: bool = False,
        ignore_args: tp.Optional[tp.Sequence[str]] = None,
        rename_args: tp.DictLike = None,
        location: tp.Optional[str] = None,
        prepend_location: tp.Optional[bool] = None,
        unpack: tp.Union[bool, str] = False,
        concat: bool = True,
        silence_warnings: bool = False,
        raise_errors: bool = False,
        execute_kwargs: tp.KwargsLike = None,
        merge_func: tp.Union[None, str, tuple, tp.Callable] = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_keys: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Run a function on data.

        Looks into the signature of the function and searches for arguments with the name `data` or
        those found among features or attributes.

        For example, the argument `open` will be substituted by `Data.open`.

        `func` can be one of the following:

        * Location to compute all indicators from. See `vectorbtpro.indicators.factory.IndicatorFactory.list_locations`.
        * Indicator name. See `vectorbtpro.indicators.factory.IndicatorFactory.get_indicator`.
        * Simulation method. See `vectorbtpro.portfolio.base.Portfolio`.
        * Any callable object
        * Iterable with any of the above. Will be stacked as columns into a DataFrame.

        Use `rename_args` to rename arguments. For example, in `vectorbtpro.portfolio.base.Portfolio`,
        data can be passed instead of `close`.

        Set `unpack` to True, "dict", or "frame" to use
        `vectorbtpro.indicators.factory.IndicatorBase.unpack`,
        `vectorbtpro.indicators.factory.IndicatorBase.to_dict`, and
        `vectorbtpro.indicators.factory.IndicatorBase.to_frame` respectively.

        Any argument in `*args` and `**kwargs` can be wrapped with `run_func_dict`/`run_arg_dict`
        to specify the value per function/argument name or index when `func` is iterable.

        Multiple function calls are executed with `vectorbtpro.utils.execution.execute`."""
        from vectorbtpro.indicators.factory import IndicatorBase, IndicatorFactory
        from vectorbtpro.indicators.talib_ import talib_func
        from vectorbtpro.portfolio.base import Portfolio

        if execute_kwargs is None:
            execute_kwargs = {}
        if merge_kwargs is None:
            merge_kwargs = {}

        _self = self
        if on_features is not None:
            _self = _self.select_features(on_features)
        if on_symbols is not None:
            _self = _self.select_symbols(on_symbols)

        if pass_as_first:
            return func(_self, *args, **kwargs)

        def _select_func_args(i, func_name, args) -> tuple:
            _args = ()
            for v in args:
                if isinstance(v, run_func_dict):
                    if func_name in v:
                        _args += (v[func_name],)
                    elif i in v:
                        _args += (v[i],)
                    elif "_def" in v:
                        _args += (v["_def"],)
                else:
                    _args += (v,)
            return _args

        def _select_func_kwargs(i, func_name, kwargs) -> dict:
            _kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, run_func_dict):
                    if func_name in v:
                        _kwargs[k] = v[func_name]
                    elif i in v:
                        _kwargs[k] = v[i]
                    elif "_def" in v:
                        _kwargs[k] = v["_def"]
                elif isinstance(v, run_arg_dict):
                    if func_name == k or i == k:
                        _kwargs.update(v)
                else:
                    _kwargs[k] = v
            return _kwargs

        if checks.is_iterable(func) and not isinstance(func, str):
            funcs_args = []
            keys = []
            for i, f in enumerate(func):
                _location = location
                if callable(f):
                    func_name = f.__name__
                elif isinstance(f, str):
                    if _location is not None:
                        func_name = f.lower().strip()
                        if prepend_location is True:
                            func_name = _location + "_" + func_name
                    else:
                        _location, f = IndicatorFactory.split_indicator_name(f)
                        if f is None:
                            raise ValueError("Sequence of locations is not supported")
                        func_name = f.lower().strip()
                        if _location is not None:
                            if prepend_location in (None, True):
                                func_name = _location + "_" + func_name
                else:
                    func_name = f
                new_args = _select_func_args(i, func_name, args)
                new_args = (_self, func_name, f, *new_args)
                new_kwargs = _select_func_kwargs(i, func_name, kwargs)
                if concat and _location == "talib_func":
                    new_kwargs["unpack_to"] = "frame"
                new_kwargs = {
                    **dict(
                        ignore_args=ignore_args,
                        rename_args=rename_args,
                        location=_location,
                        prepend_location=prepend_location,
                        unpack="frame" if concat else unpack,
                        concat=concat,
                        silence_warnings=silence_warnings,
                        raise_errors=raise_errors,
                        execute_kwargs=execute_kwargs,
                        merge_func=merge_func,
                        merge_kwargs=merge_kwargs,
                        template_context=template_context,
                        return_keys=return_keys,
                    ),
                    **new_kwargs,
                }

                funcs_args.append((self.try_run, new_args, new_kwargs))
                keys.append(str(func_name))

            keys = pd.Index(keys, name="run_func")
            results = execute(funcs_args, n_calls=len(keys), progress_desc=keys, **execute_kwargs)

            skip_indices = set()
            for i, result in enumerate(results):
                if isinstance(result, _NoResult):
                    skip_indices.add(i)
            if len(skip_indices) > 0:
                new_results = []
                keep_indices = []
                for i, result in enumerate(results):
                    if i not in skip_indices:
                        new_results.append(result)
                        keep_indices.append(i)
                results = new_results
                keys = keys[keep_indices]
            if len(results) == 0:
                raise NoResultsException

            if merge_func is None and concat:
                merge_func = "column_stack"
            if merge_func is not None:
                if isinstance(merge_func, (str, tuple)):
                    from vectorbtpro.base.merging import resolve_merge_func

                    merge_func = resolve_merge_func(merge_func)
                    merge_kwargs = {**dict(keys=keys), **merge_kwargs}
                merge_kwargs = substitute_templates(merge_kwargs, template_context, sub_id="merge_kwargs")
                if return_keys:
                    return merge_func(results, **merge_kwargs), keys
                return merge_func(results, **merge_kwargs)
            if return_keys:
                return results, keys
            return results

        if isinstance(func, str):
            func_name = func.lower().strip()
            if func_name.startswith("from_") and getattr(Portfolio, func_name):
                func = getattr(Portfolio, func_name)
                pf = func(_self, *args, **kwargs)
                if isinstance(pf, Portfolio) and unpack:
                    raise ValueError("Portfolio cannot be unpacked")
                return pf
            if location is None:
                location, func_name = IndicatorFactory.split_indicator_name(func_name)
            if location is not None and (func_name is None or func_name == "all"):
                location = location.lower().strip()
                if func_name == "all":
                    if prepend_location is None:
                        prepend_location = True
                else:
                    if prepend_location is None:
                        prepend_location = False
                if location == "talib_func":
                    indicators = IndicatorFactory.list_indicators("talib", prepend_location=False)
                else:
                    indicators = IndicatorFactory.list_indicators(location, prepend_location=False)
                return _self.run(
                    indicators,
                    *args,
                    ignore_args=ignore_args,
                    rename_args=rename_args,
                    location=location,
                    prepend_location=prepend_location,
                    unpack=unpack,
                    concat=concat,
                    silence_warnings=silence_warnings,
                    raise_errors=raise_errors,
                    execute_kwargs=execute_kwargs,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                    return_keys=return_keys,
                    **kwargs,
                )
            if location is not None:
                location = location.lower().strip()
                if location == "talib_func":
                    func = talib_func(func_name)
                else:
                    func = IndicatorFactory.get_indicator(func_name, location=location)
            else:
                func = IndicatorFactory.get_indicator(func_name)
        if isinstance(func, type) and issubclass(func, IndicatorBase):
            func = func.run

        with_kwargs = {}
        for arg_name in get_func_arg_names(func):
            real_arg_name = arg_name
            if ignore_args is not None:
                if arg_name in ignore_args:
                    continue
            if rename_args is not None:
                if arg_name in rename_args:
                    arg_name = rename_args[arg_name]
            if real_arg_name not in kwargs:
                if arg_name == "data":
                    with_kwargs[real_arg_name] = _self
                elif arg_name == "wrapper":
                    with_kwargs[real_arg_name] = _self.symbol_wrapper
                elif arg_name in ("input_shape", "shape"):
                    with_kwargs[real_arg_name] = _self.shape
                elif arg_name in ("target_shape", "shape_2d"):
                    with_kwargs[real_arg_name] = _self.shape_2d
                elif arg_name in ("input_index", "index"):
                    with_kwargs[real_arg_name] = _self.index
                elif arg_name in ("input_columns", "columns"):
                    with_kwargs[real_arg_name] = _self.columns
                elif arg_name == "freq":
                    with_kwargs[real_arg_name] = _self.freq
                elif arg_name == "hlc3":
                    with_kwargs[real_arg_name] = _self.hlc3
                elif arg_name == "ohlc4":
                    with_kwargs[real_arg_name] = _self.ohlc4
                elif arg_name == "returns":
                    with_kwargs[real_arg_name] = _self.returns
                else:
                    feature_idx = _self.get_feature_idx(arg_name)
                    if feature_idx != -1:
                        with_kwargs[real_arg_name] = _self.get_feature(feature_idx)
        new_args, new_kwargs = extend_args(func, args, kwargs, **with_kwargs)
        out = func(*new_args, **new_kwargs)
        if isinstance(unpack, bool):
            if unpack:
                if isinstance(out, IndicatorBase):
                    out = out.unpack()
        elif isinstance(unpack, str) and unpack.lower() == "dict":
            if isinstance(out, IndicatorBase):
                out = out.to_dict()
            elif isinstance(out, pd.Series):
                out = {func.__name__: out}
        elif isinstance(unpack, str) and unpack.lower() == "frame":
            if isinstance(out, IndicatorBase):
                out = out.to_frame()
            elif isinstance(out, pd.Series):
                out = out.to_frame()
        else:
            raise ValueError(f"Invalid option unpack='{unpack}'")
        return out

    # ############# Persisting ############# #

    def to_csv(
        self,
        dir_path: tp.Union[tp.PathLike, feature_dict, symbol_dict] = ".",
        ext: tp.Union[str, feature_dict, symbol_dict] = "csv",
        path_or_buf: tp.Union[None, str, feature_dict, symbol_dict] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict] = None,
        check_dict_type: bool = True,
        **kwargs,
    ) -> None:
        """Save data into CSV file(s).

        Any argument can be provided per feature using `feature_dict` or per symbol using `symbol_dict`,
        depending on the format of the data dictionary.

        Each feature/symbol gets saved to a separate file, that's why the first argument is the path
        to the directory, not file! If there's only one file, you can specify the file path via
        `path_or_buf`. If there are multiple files, use the same argument but wrap the multiple paths
        with `feature_dict`/`symbol_dict`."""
        for k, v in self.data.items():
            if check_dict_type:
                self.check_dict_type(path_or_buf, "path_or_buf")
            if path_or_buf is None:
                if check_dict_type:
                    self.check_dict_type(dir_path, "dir_path")
                if isinstance(dir_path, key_dict):
                    _dir_path = dir_path[k]
                else:
                    _dir_path = dir_path
                _dir_path = Path(_dir_path)
                if check_dict_type:
                    self.check_dict_type(ext, "ext")
                if isinstance(ext, key_dict):
                    _ext = ext[k]
                else:
                    _ext = ext
                _path_or_buf = str(Path(_dir_path) / f"{k}.{_ext}")
            elif isinstance(path_or_buf, key_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf
            if isinstance(_path_or_buf, CustomTemplate):
                _path_or_buf = _path_or_buf.substitute(dict(symbol=k, data=v), sub_id="path_or_buf")
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            sep = _kwargs.pop("sep", None)
            if isinstance(_path_or_buf, (str, Path)):
                _path_or_buf = Path(_path_or_buf)
                if check_dict_type:
                    self.check_dict_type(mkdir_kwargs, "mkdir_kwargs")
                if isinstance(mkdir_kwargs, key_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_key_kwargs(k, mkdir_kwargs, check_dict_type=check_dict_type)
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
        file_path: tp.Union[tp.PathLike, feature_dict, symbol_dict] = ".",
        key: tp.Union[None, str, feature_dict, symbol_dict] = None,
        path_or_buf: tp.Union[None, str, feature_dict, symbol_dict] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict] = None,
        format: str = "table",
        check_dict_type: bool = True,
        **kwargs,
    ) -> None:
        """Save data into an HDF file.

        Any argument can be provided per feature using `feature_dict` or per symbol using `symbol_dict`,
        depending on the format of the data dictionary.

        If `file_path` exists, and it's a directory, will create inside it a file named
        after this class. This won't work with directories that do not exist, otherwise
        they could be confused with file names."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        for k, v in self.data.items():
            if check_dict_type:
                self.check_dict_type(path_or_buf, "path_or_buf")
            if path_or_buf is None:
                if check_dict_type:
                    self.check_dict_type(file_path, "file_path")
                if isinstance(file_path, key_dict):
                    _file_path = file_path[k]
                else:
                    _file_path = file_path
                _file_path = Path(_file_path)
                if _file_path.exists() and _file_path.is_dir():
                    _file_path /= type(self).__name__ + ".h5"
                _dir_path = _file_path.parent
                if check_dict_type:
                    self.check_dict_type(mkdir_kwargs, "mkdir_kwargs")
                if isinstance(mkdir_kwargs, key_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_key_kwargs(k, mkdir_kwargs, check_dict_type=check_dict_type)
                check_mkdir(_dir_path, **_mkdir_kwargs)
                _path_or_buf = str(_file_path)
            elif isinstance(path_or_buf, key_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf
            if isinstance(_path_or_buf, CustomTemplate):
                _path_or_buf = _path_or_buf.substitute(dict(symbol=k, data=v), sub_id="path_or_buf")
            if check_dict_type:
                self.check_dict_type(key, "key")
            if key is None:
                _key = str(k)
            elif isinstance(key, key_dict):
                _key = key[k]
            else:
                _key = key
            if isinstance(_key, CustomTemplate):
                _key = _key.substitute(dict(symbol=k, data=v), sub_id="key")
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            v.to_hdf(path_or_buf=_path_or_buf, key=_key, format=format, **_kwargs)

    @classmethod
    def from_csv(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Use `CSVData` to load data from CSV and switch the class back to this class.

        Use `fetch_kwargs` to provide keyword arguments that were originally used in fetching."""
        from vectorbtpro.data.custom.csv import CSVData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = CSVData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    @classmethod
    def from_hdf(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Use `HDFData` to load data from HDF and switch the class back to this class.

        Use `fetch_kwargs` to provide keyword arguments that were originally used in fetching."""
        from vectorbtpro.data.custom.hdf import HDFData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        if len(args) == 0 and "keys" not in kwargs and "features" not in kwargs and "symbols" not in kwargs:
            args = (cls.__name__ + ".h5",)
        data = HDFData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

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
            total_features=dict(
                title="Total Features",
                check_is_feature_oriented=True,
                calc_func=lambda self: len(self.features),
                agg_func=None,
                tags="data",
            ),
            total_symbols=dict(
                title="Total Symbols",
                check_is_symbol_oriented=True,
                calc_func=lambda self: len(self.symbols),
                tags="data",
            ),
            null_counts=dict(
                title="Null Counts",
                calc_func=lambda self, group_by: {
                    k: v.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by) for k, v in self.data.items()
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
        column: tp.Optional[tp.Hashable] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        feature_map: tp.KwargsLike = None,
        plot_volume: tp.Optional[bool] = None,
        base: tp.Optional[float] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot either one feature of multiple symbols, or OHLC(V) of one symbol.

        Args:
            column (hashable): Name of the feature or symbol to plot.

                Depends on the data orientation.
            feature (hashable): Name of the feature to plot.
            symbol (hashable): Name of the symbol to plot.
            feature_map (sequence of str): Dictionary mapping the feature names to OHLCV.

                Applied only if OHLC(V) is plotted.
            plot_volume (bool): Whether to plot volume beneath.

                Applied only if OHLC(V) is plotted.
            base (float): Rebase all series of a feature to a given initial base.

                !!! note
                    The feature must contain prices.

                Applied only if lines are plotted.
            kwargs (dict): Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`
                for lines and to `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot` for OHLC(V).

        Usage:
            * Plot the lines of one feature across all symbols:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> start = '2021-01-01 UTC'  # crypto is in UTC
            >>> end = '2021-06-01 UTC'
            >>> data = vbt.YFData.pull(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
            ```

            [=100% "100%"]{: .candystripe}

            ```pycon
            >>> data.plot(feature='Close', base=1).show()
            ```

            * Plot OHLC(V) of one symbol (only if data contains the respective features):

            ![](/assets/images/api/data_plot.svg){: .iimg loading=lazy }

            ```pycon
            >>> data.plot(symbol='BTC-USD').show()
            ```

            ![](/assets/images/api/data_plot_ohlcv.svg){: .iimg loading=lazy }
        """
        if column is not None:
            if self.feature_oriented:
                if symbol is not None:
                    raise ValueError("Either column or symbol can be provided, not both")
                symbol = column
            else:
                if feature is not None:
                    raise ValueError("Either column or feature can be provided, not both")
                feature = column
        if feature is None and self.has_ohlc:
            data = self.get(symbols=symbol, squeeze_symbols=True)
            if isinstance(data, tuple):
                raise ValueError("Cannot plot OHLC of multiple symbols. Select one symbol.")
            return data.vbt.ohlcv(feature_map=feature_map).plot(plot_volume=plot_volume, **kwargs)
        data = self.get(features=feature, symbols=symbol, squeeze_features=True, squeeze_symbols=True)
        if isinstance(data, tuple):
            raise ValueError("Cannot plot multiple features and symbols. Select one feature or symbol.")
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

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=RepEval(
                """
                if symbols is None:
                    symbols = self.symbols
                if not self.has_multiple_keys(symbols):
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
    def build_feature_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build feature config documentation."""
        if source_cls is None:
            source_cls = Data
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "feature_config").__doc__)).substitute(
            {"feature_config": cls.feature_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_feature_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Data.feature_config`."""
        __pdoc__[cls.__name__ + ".feature_config"] = cls.build_feature_config_doc(source_cls=source_cls)


Data.override_feature_config_doc(__pdoc__)
Data.override_metrics_doc(__pdoc__)
Data.override_subplots_doc(__pdoc__)
