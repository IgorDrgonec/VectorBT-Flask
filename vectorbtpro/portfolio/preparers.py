# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for preparing portfolio simulations."""

import inspect
import string
import warnings
from collections import defaultdict
from collections import namedtuple
from datetime import timedelta, time
from functools import cached_property as cachedproperty
from pathlib import Path

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import index_dict
from vectorbtpro.base.reshaping import BCO, Default, Ref
from vectorbtpro.base.reshaping import (
    broadcast_array_to,
    broadcast,
)
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio import nb, enums
from vectorbtpro.portfolio.call_seq import require_call_seq, build_call_seq
from vectorbtpro.portfolio.decorators import override_arg_config, attach_arg_properties
from vectorbtpro.portfolio.orders import FSOrders
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks, chunking as ch
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.cutting import suggest_module_path, cut_and_save_func
from vectorbtpro.utils.datetime_ import (
    freq_to_timedelta64,
    parse_timedelta,
    time_to_timedelta,
    try_align_to_datetime_index,
)
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.mapping import to_mapping
from vectorbtpro.utils.module_ import import_module_from_path
from vectorbtpro.utils.params import Param
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "PrepResult",
    "BasePreparer",
    "FOPreparer",
    "FSPreparer",
]

__pdoc__ = {}


PrepResultT = tp.TypeVar("PrepResultT", bound="PrepResult")


class PrepResult(Configured):
    """Result of preparation."""

    def __init__(
        self,
        sim_func: tp.Optional[tp.Callable] = None,
        sim_args: tp.Optional[tp.Kwargs] = None,
        pf_args: tp.Optional[tp.Kwargs] = None,
    ) -> None:
        Configured.__init__(self, sim_func=sim_func, sim_args=sim_args, pf_args=pf_args)

    @cachedproperty
    def sim_func(self) -> tp.Optional[tp.Callable]:
        """Simulation function."""
        return self.config["sim_func"]

    @cachedproperty
    def sim_args(self) -> tp.Kwargs:
        """Simulation arguments."""
        return self.config["sim_args"]

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        """Portfolio arguments."""
        return self.config["pf_args"]


base_arg_config = ReadonlyConfig(
    dict(
        data=dict(),
        open=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        high=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        low=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        bm_close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        cash_earnings=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        init_cash=dict(map_enum_kwargs=dict(enum=enums.InitCashMode, look_for_type=str)),
        init_position=dict(),
        init_price=dict(),
        cash_deposits=dict(),
        group_by=dict(),
        cash_sharing=dict(),
        freq=dict(),
        call_seq=dict(map_enum_kwargs=dict(enum=enums.CallSeqType, look_for_type=str)),
        attach_call_seq=dict(),
        in_outputs=dict(has_default=False),
        broadcast_named_args=dict(is_dict=True),
        broadcast_kwargs=dict(is_dict=True),
        template_context=dict(is_dict=True),
        seed=dict(),
        jitted=dict(),
        chunked=dict(),
    )
)
"""_"""

__pdoc__[
    "base_arg_config"
] = f"""Argument config for `BasePreparer`.

```python
{base_arg_config.prettify()}
```
"""


class MetaArgs(type):
    """Meta class that exposes a read-only class property `MetaArgs.arg_config`."""

    @property
    def arg_config(cls) -> Config:
        """Argument config."""
        return cls._arg_config


@attach_arg_properties
@override_arg_config(base_arg_config)
class BasePreparer(Configured, metaclass=MetaArgs):
    """Base class for preparing simulations.

    !!! warning
        Most properties are force-cached - create a new instance to override any attribute."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_arg_config"}

    _setting_keys: tp.SettingsKeys = "portfolio"

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

        # Copy writeable attrs
        self._subplots = type(self)._arg_config.copy()

    _arg_config: tp.ClassVar[Config] = HybridConfig()

    @property
    def arg_config(self) -> Config:
        """Argument config of `${cls_name}`.

        ```python
        ${arg_config}
        ```
        """
        return self._arg_config

    @classmethod
    def map_enum_value(cls, value: tp.ArrayLike, look_for_type: tp.Optional[type] = None, **kwargs) -> tp.ArrayLike:
        """Map enumerated value(s)."""
        if look_for_type is not None:
            if isinstance(value, look_for_type):
                return map_enum_fields(value, **kwargs)
            return value
        if isinstance(value, (CustomTemplate, Ref)):
            return value
        if isinstance(value, (Param, BCO, Default)):
            attr_dct = attr.asdict(value)
            if isinstance(value, Param) and attr_dct["map_template"] is None:
                attr_dct["map_template"] = RepFunc(lambda values: cls.map_enum_value(values, **kwargs))
            elif not isinstance(value, Param):
                attr_dct["value"] = cls.map_enum_value(attr_dct["value"], **kwargs)
            return type(value)(**attr_dct)
        if isinstance(value, index_dict):
            return index_dict({k: cls.map_enum_value(v, **kwargs) for k, v in value.items()})
        return map_enum_fields(value, **kwargs)

    def prepare_td_obj(self, td_obj: object) -> object:
        """Prepare a timedelta object for broadcasting."""
        if isinstance(td_obj, (str, timedelta, pd.DateOffset, pd.Timedelta)):
            td_obj = freq_to_timedelta64(td_obj)
        elif isinstance(td_obj, pd.Index):
            td_obj = td_obj.values
        return td_obj

    def prepare_dt_obj(self, dt_obj: object, ns_ago: int = 0) -> object:
        """Prepare a datetime object for broadcasting."""
        if isinstance(dt_obj, (str, time, timedelta, pd.DateOffset, pd.Timedelta)):
            dt_obj_dt_template = RepEval(
                "try_align_to_datetime_index([dt_obj], wrapper.index).vbt.to_ns() - ns_ago",
                context=dict(try_align_to_datetime_index=try_align_to_datetime_index, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            dt_obj_td_template = RepEval(
                "wrapper.index.vbt.to_period_ns(parse_timedelta(dt_obj)) - ns_ago",
                context=dict(parse_timedelta=parse_timedelta, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            dt_obj_time_template = RepEval(
                '(wrapper.index.floor("1d") + time_to_timedelta(dt_obj)).vbt.to_ns() - ns_ago',
                context=dict(time_to_timedelta=time_to_timedelta, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            if isinstance(dt_obj, str):
                try:
                    time.fromisoformat(dt_obj)
                    dt_obj = dt_obj_time_template
                except Exception as e:
                    try:
                        parse_timedelta(dt_obj)
                        dt_obj = dt_obj_td_template
                    except Exception as e:
                        dt_obj = dt_obj_dt_template
            elif isinstance(dt_obj, time):
                dt_obj = dt_obj_time_template
            else:
                dt_obj = dt_obj_td_template
        elif isinstance(dt_obj, pd.Index):
            dt_obj = dt_obj.values
        return dt_obj

    def get_raw_arg_default(self, arg_name: str, is_dict: bool = False) -> tp.Any:
        """Get raw argument default."""
        value = self.get_setting(arg_name)
        if is_dict and value is None:
            return {}
        return value

    def get_raw_arg(self, arg_name: str, is_dict: bool = False, has_default: bool = True) -> tp.Any:
        """Get raw argument."""
        value = self.config.get(arg_name, None)
        if is_dict:
            if has_default:
                return merge_dicts(self.get_raw_arg_default(arg_name), value)
            if value is None:
                return {}
            return value
        if value is None and has_default:
            return self.get_raw_arg_default(arg_name)
        return value

    def get_mapped_arg_default(self, arg_name: str, is_dict: bool = False, **kwargs) -> tp.Any:
        """Get mapped argument default."""
        raw_arg_default = self.get_raw_arg_default(arg_name, is_dict=is_dict)
        return self.map_enum_value(raw_arg_default, **kwargs)

    def get_mapped_arg(self, arg_name: str, is_dict: bool = False, has_default: bool = True, **kwargs) -> tp.Any:
        """Get mapped argument."""
        raw_arg = self.get_raw_arg(arg_name, is_dict=is_dict, has_default=has_default)
        return self.map_enum_value(raw_arg, **kwargs)

    def get_arg_default(self, arg_name: str) -> tp.Any:
        """Get argument default according to the argument config."""
        arg_config = self.arg_config[arg_name]
        if "map_enum_kwargs" in arg_config:
            arg = self.get_mapped_arg_default(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                **arg_config["map_enum_kwargs"],
            )
        else:
            arg = self.get_raw_arg_default(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
            )
        if arg_config.get("is_td", False):
            arg = self.prepare_td_obj(arg)
        if arg_config.get("is_dt", False):
            arg = self.prepare_dt_obj(arg, ns_ago=arg_config.get("ns_ago", 0))
        return arg

    def get_arg(self, arg_name: str) -> tp.Any:
        """Get mapped argument according to the argument config."""
        arg_config = self.arg_config[arg_name]
        if "map_enum_kwargs" in arg_config:
            arg = self.get_mapped_arg(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                has_default=arg_config.get("has_default", True),
                **arg_config["map_enum_kwargs"],
            )
        else:
            arg = self.get_raw_arg(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                has_default=arg_config.get("has_default", True),
            )
        if arg_config.get("is_td", False):
            arg = self.prepare_td_obj(arg)
        if arg_config.get("is_dt", False):
            arg = self.prepare_dt_obj(arg, ns_ago=arg_config.get("ns_ago", 0))
        return arg

    def __getitem__(self, arg_name) -> tp.Any:
        return self.get_arg(arg_name)

    def td_arr_to_ns(self, td_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Prepare a timedelta array."""
        if td_arr.dtype == object:
            if td_arr.ndim in (0, 1):
                td_arr = pd.to_timedelta(td_arr)
                if isinstance(td_arr, pd.Timedelta):
                    td_arr = td_arr.to_timedelta64()
                else:
                    td_arr = td_arr.values
            else:
                td_arr_cols = []
                for col in range(td_arr.shape[1]):
                    td_arr_col = pd.to_timedelta(td_arr[:, col])
                    td_arr_cols.append(td_arr_col.values)
                td_arr = np.column_stack(td_arr_cols)
        return td_arr.astype(np.int64)

    def dt_arr_to_ns(self, dt_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Prepare a datetime array."""
        if dt_arr.dtype == object:
            if dt_arr.ndim in (0, 1):
                dt_arr = pd.to_datetime(dt_arr).tz_localize(None)
                if isinstance(dt_arr, pd.Timestamp):
                    dt_arr = dt_arr.to_datetime64()
                else:
                    dt_arr = dt_arr.values
            else:
                dt_arr_cols = []
                for col in range(dt_arr.shape[1]):
                    dt_arr_col = pd.to_datetime(dt_arr[:, col]).tz_localize(None)
                    dt_arr_cols.append(dt_arr_col.values)
                dt_arr = np.column_stack(dt_arr_cols)
        return dt_arr.astype(np.int64)

    def prepare_post_arg(self, arg_name: str, value: tp.Optional[tp.ArrayLike] = None) -> tp.ArrayLike:
        """Prepare an argument after broadcasting."""
        if value is None:
            arg = self.post_args[arg_name]
        else:
            arg = value
        if arg is not None:
            arg_config = self.arg_config[arg_name]
            if "map_enum_kwargs" in arg_config:
                arg = map_enum_fields(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.td_arr_to_ns(arg)
            if arg_config.get("is_dt", False):
                arg = self.dt_arr_to_ns(arg)
            if "subdtype" in arg_config:
                checks.assert_subdtype(arg, arg_config["subdtype"], arg_name=arg_name)
        return arg

    # ############# Ready arguments ############# #

    @cachedproperty
    def init_cash_mode(self) -> tp.Optional[int]:
        """Initial cash mode."""
        init_cash = self["init_cash"]
        if init_cash in enums.InitCashMode:
            return init_cash
        return None

    @cachedproperty
    def group_by(self) -> tp.GroupByLike:
        """Argument `group_by`."""
        group_by = self["group_by"]
        if group_by is None and self.cash_sharing:
            return True
        return group_by

    @cachedproperty
    def auto_call_seq(self) -> bool:
        """Whether automatic call sequence is enabled."""
        call_seq = self["call_seq"]
        return checks.is_int(call_seq) and call_seq == enums.CallSeqType.Auto

    def set_seed(self) -> None:
        """Set seed."""
        seed = self.seed
        if seed is not None:
            set_seed(seed)

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre_open(self) -> tp.ArrayLike:
        """Argument `open` before broadcasting."""
        open = self["open"]
        if open is None:
            if self.data is not None:
                open = self.data.open
            if open is None:
                return np.nan
        return open

    @cachedproperty
    def pre_high(self) -> tp.ArrayLike:
        """Argument `high` before broadcasting."""
        high = self["high"]
        if high is None:
            if self.data is not None:
                high = self.data.high
            if high is None:
                return np.nan
        return high

    @cachedproperty
    def pre_low(self) -> tp.ArrayLike:
        """Argument `low` before broadcasting."""
        low = self["low"]
        if low is None:
            if self.data is not None:
                low = self.data.low
            if low is None:
                return np.nan
        return low

    @cachedproperty
    def pre_close(self) -> tp.ArrayLike:
        """Argument `close` before broadcasting."""
        close = self["close"]
        if close is None:
            if self.data is not None:
                close = self.data.close
            if close is None:
                return np.nan
        return close

    @cachedproperty
    def pre_bm_close(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `bm_close` before broadcasting."""
        bm_close = self["bm_close"]
        if bm_close is not None and not isinstance(bm_close, bool):
            return bm_close
        return None

    @cachedproperty
    def pre_init_cash(self) -> tp.ArrayLike:
        """Argument `init_cash` before broadcasting."""
        if self.init_cash_mode is not None:
            return np.inf
        return self["init_cash"]

    @cachedproperty
    def pre_init_position(self) -> tp.ArrayLike:
        """Argument `init_position` before broadcasting."""
        return self["init_position"]

    @cachedproperty
    def pre_init_price(self) -> tp.ArrayLike:
        """Argument `init_price` before broadcasting."""
        return self["init_price"]

    @cachedproperty
    def pre_cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits` before broadcasting."""
        return self["cash_deposits"]

    @cachedproperty
    def pre_freq(self) -> tp.Optional[tp.FrequencyLike]:
        """Argument `freq` before casting to nanosecond format."""
        freq = self["freq"]
        if freq is None and self.data is not None:
            return self.data.freq
        return freq

    @cachedproperty
    def pre_call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq` before broadcasting."""
        if self.auto_call_seq:
            return None
        return self["call_seq"]

    @cachedproperty
    def pre_in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Argument `in_outputs` before broadcasting."""
        in_outputs = self["in_outputs"]
        if (
            in_outputs is not None
            and not isinstance(in_outputs, CustomTemplate)
            and not checks.is_namedtuple(in_outputs)
        ):
            in_outputs = to_mapping(in_outputs)
            in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        return in_outputs

    @cachedproperty
    def pre_template_context(self) -> tp.Kwargs:
        """Argument `template_context` before broadcasting."""
        return merge_dicts(dict(preparer=self), self["template_context"])

    # ############# Broadcasting ############# #

    @cachedproperty
    def pre_args(self) -> tp.Kwargs:
        """Arguments before broadcasting."""
        pre_args = dict()
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                pre_args[k] = getattr(self, "pre_" + k)
        return pre_args

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        """Default keyword arguments for broadcasting."""
        return dict(
            to_pd=False,
            keep_flex=True,
            wrapper_kwargs=dict(
                freq=self.pre_freq,
                group_by=self.group_by,
            ),
            return_wrapper=True,
        )

    @cachedproperty
    def broadcast_kwargs(self) -> tp.Kwargs:
        """Argument `broadcast_kwargs`."""
        arg_broadcast_kwargs = defaultdict(dict)
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                broadcast_kwargs = v.get("broadcast_kwargs", None)
                if broadcast_kwargs is None:
                    broadcast_kwargs = {}
                for k2, v2 in broadcast_kwargs.items():
                    arg_broadcast_kwargs[k2][k] = v2
        return merge_dicts(
            self.def_broadcast_kwargs,
            dict(arg_broadcast_kwargs),
            self["broadcast_kwargs"],
        )

    @cachedproperty
    def broadcast_result(self) -> tp.Any:
        """Result of broadcasting."""
        args_to_broadcast = merge_dicts(self.pre_args, self.broadcast_named_args)
        return broadcast(args_to_broadcast, **self.broadcast_kwargs)

    @cachedproperty
    def post_args(self) -> tp.Kwargs:
        """Arguments after broadcasting."""
        return self.broadcast_result[0]

    @cachedproperty
    def post_broadcast_named_args(self) -> tp.Kwargs:
        """Custom arguments after broadcasting."""
        broadcast_named_args = self.broadcast_named_args
        if broadcast_named_args is None:
            return dict()
        post_broadcast_named_args = dict()
        for k, v in self.post_args.items():
            if k in broadcast_named_args:
                post_broadcast_named_args[k] = v
        return post_broadcast_named_args

    @cachedproperty
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self.broadcast_result[1]

    @cachedproperty
    def index(self) -> tp.Array1d:
        """Index in nanosecond format."""
        return self.wrapper.ns_index

    @cachedproperty
    def freq(self) -> int:
        """Frequency in nanosecond format."""
        return self.wrapper.ns_freq

    # ############# After broadcasting ############# #

    @cachedproperty
    def target_shape(self) -> tp.Shape:
        """Target shape."""
        return self.wrapper.shape_2d

    @cachedproperty
    def cs_group_lens(self) -> tp.GroupLens:
        """Cash sharing aware group lengths."""
        cs_group_lens = self.wrapper.grouper.get_group_lens(group_by=None if self.cash_sharing else False)
        checks.assert_subdtype(cs_group_lens, np.integer, arg_name="cs_group_lens")
        return cs_group_lens

    @cachedproperty
    def group_lens(self) -> tp.GroupLens:
        """Group lengths."""
        return self.wrapper.grouper.get_group_lens(group_by=self.group_by)

    @cachedproperty
    def init_cash(self) -> tp.ArrayLike:
        """Argument `init_cash`."""
        init_cash = broadcast_array_to(self.pre_init_cash, len(self.cs_group_lens))
        checks.assert_subdtype(init_cash, np.number, arg_name="init_cash")
        init_cash = np.require(init_cash, dtype=np.float_)
        return init_cash

    @cachedproperty
    def init_position(self) -> tp.ArrayLike:
        """Argument `init_position`."""
        init_position = broadcast_array_to(self.pre_init_position, self.target_shape[1])
        checks.assert_subdtype(init_position, np.number, arg_name="init_position")
        init_position = np.require(init_position, dtype=np.float_)
        if (((init_position > 0) | (init_position < 0)) & np.isnan(self.init_price)).any():
            warnings.warn(f"Initial position has undefined price. Set init_price.", stacklevel=2)
        return init_position

    @cachedproperty
    def init_price(self) -> tp.ArrayLike:
        """Argument `init_price`."""
        init_price = broadcast_array_to(self.pre_init_price, self.target_shape[1])
        checks.assert_subdtype(init_price, np.number, arg_name="init_price")
        return np.require(init_price, dtype=np.float_)

    @cachedproperty
    def cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits`."""
        cash_deposits = self["cash_deposits"]
        checks.assert_subdtype(cash_deposits, np.number, arg_name="cash_deposits")
        return broadcast(
            cash_deposits,
            to_shape=(self.target_shape[0], len(self.cs_group_lens)),
            to_pd=False,
            keep_flex=True,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=self.broadcast_kwargs.get("require_kwargs", {}),
        )

    @cachedproperty
    def call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq`."""
        call_seq = self.pre_call_seq
        if call_seq is None and self.attach_call_seq:
            call_seq = enums.CallSeqType.Default
        if call_seq is not None:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(broadcast(call_seq, to_shape=self.target_shape, to_pd=False))
            else:
                call_seq = build_call_seq(self.target_shape, self.group_lens, call_seq_type=call_seq)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer, arg_name="call_seq")
        return call_seq

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        """Argument `template_context`."""
        builtin_args = {}
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                builtin_args[k] = getattr(self, k)
        return merge_dicts(
            dict(
                target_shape=self.target_shape,
                group_lens=self.group_lens,
                cs_group_lens=self.cs_group_lens,
                cash_sharing=self.cash_sharing,
                index=self.index,
                freq=self.freq,
                init_cash=self.init_cash,
                init_position=self.init_position,
                init_price=self.init_price,
                cash_deposits=self.cash_deposits,
                call_seq=self.call_seq,
                auto_call_seq=self.auto_call_seq,
                attach_call_seq=self.attach_call_seq,
                in_outputs=self.pre_in_outputs,
                wrapper=self.wrapper,
            ),
            builtin_args,
            self.post_broadcast_named_args,
            self.pre_template_context,
        )

    @cachedproperty
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Argument `in_outputs`."""
        return substitute_templates(self.pre_in_outputs, self.template_context, sub_id="in_outputs")

    # ############# Result ############# #

    @cachedproperty
    def sim_func(self) -> tp.Optional[tp.Callable]:
        """Simulation function."""
        return None

    @cachedproperty
    def sim_arg_map(self) -> tp.Kwargs:
        """Map of the simulation arguments to the preparer attributes."""
        return dict()

    @cachedproperty
    def sim_args(self) -> tp.Optional[tp.Kwargs]:
        """Arguments to be passed to the simulation."""
        if self.sim_func is not None:
            sim_arg_map = self.sim_arg_map
            func_arg_names = get_func_arg_names(self.sim_func)
            sim_args = {}
            for k in func_arg_names:
                arg_attr = sim_arg_map.get(k, k)
                if arg_attr is not None:
                    sim_args[k] = getattr(self, arg_attr)
            return sim_args
        return None

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        """Arguments to be passed to the portfolio."""
        kwargs = dict()
        for k, v in self.config.items():
            if k not in self.arg_config:
                kwargs[k] = v
        return dict(
            wrapper=self.wrapper,
            open=self.open if self.pre_open is not np.nan else None,
            high=self.high if self.pre_high is not np.nan else None,
            low=self.low if self.pre_low is not np.nan else None,
            close=self.close,
            cash_sharing=self.cash_sharing,
            init_cash=self.init_cash if self.init_cash_mode is None else self.init_cash_mode,
            init_position=self.init_position,
            init_price=self.init_price,
            bm_close=self.bm_close,
            **kwargs,
        )

    @cachedproperty
    def result(self) -> PrepResult:
        """Result as an instance of `PrepResult`."""
        return PrepResult(sim_func=self.sim_func, sim_args=self.sim_args, pf_args=self.pf_args)

    # ############# Docs ############# #

    @classmethod
    def build_arg_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build argument config documentation."""
        if source_cls is None:
            source_cls = BasePreparer
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "arg_config").__doc__)).substitute(
            {"arg_config": cls.arg_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_arg_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `BasePreparer.arg_config`."""
        __pdoc__[cls.__name__ + ".arg_config"] = cls.build_arg_config_doc(source_cls=source_cls)


BasePreparer.override_arg_config_doc(__pdoc__)

fo_arg_config = ReadonlyConfig(
    dict(
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.SizeType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.SizeType.Amount)),
        ),
        direction=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.Direction),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.Direction.Both)),
        ),
        fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        fixed_fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        slippage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        min_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        max_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_granularity=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        leverage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=1.0)),
        ),
        leverage_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LeverageMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LeverageMode.Lazy)),
        ),
        reject_prob=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        price_area_vio_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceAreaVioMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceAreaVioMode.Ignore)),
        ),
        allow_partial=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=True)),
        ),
        raise_reject=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        log=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        max_orders=dict(),
        max_logs=dict(),
    )
)
"""_"""

__pdoc__[
    "fo_arg_config"
] = f"""Argument config for `FOPreparer`.

```python
{fo_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fo_arg_config)
class FOPreparer(BasePreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_orders`."""

    _setting_keys: tp.SettingsKeys = "portfolio.from_orders"

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre_from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting."""
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def pre_max_orders(self) -> tp.Optional[int]:
        """Argument `max_orders` before broadcasting."""
        return self["max_orders"]

    @cachedproperty
    def pre_max_logs(self) -> tp.Optional[int]:
        """Argument `max_logs` before broadcasting."""
        return self["max_logs"]

    # ############# After broadcasting ############# #

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Arguments `price` and `from_ago` after broadcasting."""
        price = self.post_price
        from_ago = self.post_from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                if next_open_mask.any() or next_close_mask.any():
                    price = price.astype(np.float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=np.int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Argument `price`."""
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago`."""
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_orders(self) -> tp.Optional[int]:
        """Argument `max_orders`."""
        max_orders = self.pre_max_orders
        if max_orders is None:
            _size = self.post_size
            if _size.size == 1:
                max_orders = self.target_shape[0] * int(not np.isnan(_size.item(0)))
            else:
                if _size.shape[0] == 1 and self.target_shape[0] > 1:
                    max_orders = self.target_shape[0] * int(np.any(~np.isnan(_size)))
                else:
                    max_orders = int(np.max(np.sum(~np.isnan(_size), axis=0)))
        return max_orders

    @cachedproperty
    def max_logs(self) -> tp.Optional[int]:
        """Argument `max_logs`."""
        max_logs = self.pre_max_logs
        if max_logs is None:
            _log = self.post_log
            if _log.size == 1:
                max_logs = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_logs = self.target_shape[0] * int(np.any(_log))
                else:
                    max_logs = int(np.max(np.sum(_log, axis=0)))
        return max_logs

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                group_lens=self.group_lens if self.flexible_mode else self.cs_group_lens,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_orders=self.max_orders,
                max_logs=self.max_logs,
            ),
            BasePreparer.template_context.func(self),
        )

    # ############# Result ############# #

    @cachedproperty
    def sim_func(self) -> tp.Optional[tp.Callable]:
        func = jit_reg.resolve_option(nb.from_orders_nb, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def sim_arg_map(self) -> tp.Kwargs:
        return dict(group_lens="cs_group_lens")


FOPreparer.override_arg_config_doc(__pdoc__)


def adapt_staticized_to_udf(staticized: tp.Kwargs, func: tp.Union[str, tp.Callable], func_name: str) -> None:
    """Adapt `staticized` dictionary to a UDF."""
    sim_func_module = inspect.getmodule(staticized["func"])
    if isinstance(func, (str, Path)):
        if isinstance(func, str) and not func.endswith(".py") and hasattr(sim_func_module, func):
            staticized[f"{func_name}_block"] = func
            return None
        func = Path(func)
        module_path = func.resolve()
    else:
        if inspect.getmodule(func) == sim_func_module:
            staticized[f"{func_name}_block"] = func.__name__
            return None
        module = inspect.getmodule(func)
        if not hasattr(module, "__file__"):
            raise TypeError(f"{func_name} must be defined in a Python file")
        module_path = Path(module.__file__).resolve()
    if "import_lines" not in staticized:
        staticized["import_lines"] = []
    reload = staticized.get("reload", False)
    staticized["import_lines"].extend(
        [
            f'{func_name}_path = r"{module_path}"',
            f"globals().update(vbt.import_module_from_path({func_name}_path).__dict__, reload={reload})",
        ]
    )


def resolve_dynamic_simulator(simulator_name: str, staticized: tp.KwargsLike) -> tp.Callable:
    """Resolve a dynamic simulator."""
    if staticized is None:
        func = getattr(nb, simulator_name)
    else:
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            module_path = suggest_module_path(
                staticized.get("suggest_fname", simulator_name),
                path=staticized.pop("path", None),
                mkdir_kwargs=staticized.get("mkdir_kwargs", None),
            )
            if "new_func_name" not in staticized:
                staticized["new_func_name"] = simulator_name

            if staticized.pop("override", False) or not module_path.exists():
                if "skip_func" not in staticized:

                    def _skip_func(out_lines, func_name):
                        to_skip = lambda x: f"def {func_name}" in x or x.startswith(f"{func_name}_path =")
                        return any(map(to_skip, out_lines))

                    staticized["skip_func"] = _skip_func
                module_path = cut_and_save_func(path=module_path, **staticized)
            reload = staticized.pop("reload", False)
            module = import_module_from_path(module_path, reload=reload)
            func = getattr(module, staticized["new_func_name"])
        else:
            func = staticized
    return func


fs_arg_config = ReadonlyConfig(
    dict(
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        direction=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.Direction),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.Direction.Both)),
        ),
        adjust_func_nb=dict(),
        adjust_args=dict(),
        signal_func_nb=dict(),
        signal_args=dict(),
        post_segment_func_nb=dict(),
        post_segment_args=dict(),
        order_mode=dict(),
        size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.SizeType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.SizeType.Amount)),
        ),
        fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        fixed_fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        slippage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        min_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        max_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_granularity=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        leverage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=1.0)),
        ),
        leverage_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LeverageMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LeverageMode.Lazy)),
        ),
        reject_prob=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        price_area_vio_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceAreaVioMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceAreaVioMode.Ignore)),
        ),
        allow_partial=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=True)),
        ),
        raise_reject=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        log=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        accumulate=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.AccumulationMode, ignore_type=(int, bool)),
            subdtype=(np.integer, np.bool_),
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.AccumulationMode.Disabled)),
        ),
        upon_long_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_short_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_dir_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DirectionConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.DirectionConflictMode.Ignore)),
        ),
        upon_opposite_entry=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OppositeEntryMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OppositeEntryMode.ReverseReduce)),
        ),
        order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        limit_tif=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_expiry=dict(
            broadcast=True,
            is_dt=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_reverse=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        upon_adj_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepIgnore)),
        ),
        upon_opp_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.CancelExecute)),
        ),
        use_stops=dict(),
        stop_ladder=dict(map_enum_kwargs=dict(enum=enums.StopLadderMode, look_for_type=str)),
        sl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_th=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tp_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        td_stop=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        dt_stop=dict(
            broadcast=True,
            is_dt=True,
            ns_ago=1,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        stop_entry_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopEntryPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopEntryPrice.Close)),
        ),
        stop_exit_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitPrice.Stop)),
        ),
        stop_exit_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitType.Close)),
        ),
        stop_order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        stop_limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        upon_stop_update=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopUpdateMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopUpdateMode.Override)),
        ),
        upon_adj_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)),
        ),
        upon_opp_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)),
        ),
        delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.DeltaFormat.Percent)),
        ),
        time_delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.TimeDeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.TimeDeltaFormat.Index)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        fill_pos_info=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        max_orders=dict(),
        max_logs=dict(),
        staticized=dict(),
    )
)
"""_"""

__pdoc__[
    "fs_arg_config"
] = f"""Argument config for `FSPreparer`.

```python
{fs_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fs_arg_config)
class FSPreparer(BasePreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_signals`."""

    _setting_keys: tp.SettingsKeys = "portfolio.from_signals"

    # ############# Mode resolution ############# #

    @cachedproperty
    def pre_staticized(self) -> tp.StaticizedOption:
        """Argument `staticized` before its resolution."""
        staticized = self["staticized"]
        if isinstance(staticized, bool):
            if staticized:
                staticized = dict()
            else:
                staticized = None
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if "func" not in staticized:
                staticized["func"] = nb.from_signal_func_nb
        return staticized

    @cachedproperty
    def order_mode(self) -> bool:
        """Argument `order_mode`."""
        order_mode = self["order_mode"]
        if order_mode is None:
            order_mode = False
        return order_mode

    @cachedproperty
    def flexible_mode(self) -> tp.StaticizedOption:
        """Whether the flexible mode is enabled."""
        return (
            self["adjust_func_nb"] is not None
            or self["signal_func_nb"] is not None
            or self["post_segment_func_nb"] is not None
            or self.order_mode
            or self.pre_staticized is not None
        )

    @cachedproperty
    def pre_ls_mode(self) -> bool:
        """Whether direction-aware mode is enabled before resolution."""
        return self["short_entries"] is not None or self["short_exits"] is not None

    @cachedproperty
    def pre_signals_mode(self) -> bool:
        """Whether signals mode is enabled before resolution."""
        return self["entries"] is not None or self["exits"] is not None or self.pre_ls_mode

    @cachedproperty
    def ls_mode(self) -> bool:
        """Whether direction-aware mode is enabled."""
        if not self.pre_signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        ls_mode = self.pre_ls_mode
        if self.config.get("direction", None) is not None and ls_mode:
            raise ValueError("Direction and short signal arrays cannot be used together")
        return ls_mode

    @cachedproperty
    def signals_mode(self) -> bool:
        """Whether signals mode is enabled."""
        if not self.pre_signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        signals_mode = self.pre_signals_mode
        if signals_mode and self.order_mode:
            raise ValueError("Signal arrays and order mode cannot be used together")
        return signals_mode

    @cachedproperty
    def signal_func_mode(self) -> bool:
        """Whether signal function mode is enabled."""
        return self.flexible_mode and not self.signals_mode and not self.order_mode

    @cachedproperty
    def adjust_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `adjust_func_nb`."""
        if self.flexible_mode:
            if self["adjust_func_nb"] is None:
                return nb.no_adjust_func_nb
            return self["adjust_func_nb"]
        return None

    @cachedproperty
    def signal_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `signal_func_nb`."""
        if self.flexible_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return nb.ls_signal_func_nb
                if self.signals_mode:
                    return nb.dir_signal_func_nb
                if self.order_mode:
                    return nb.order_signal_func_nb
                return None
            return self["signal_func_nb"]
        return None

    @cachedproperty
    def post_segment_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `post_segment_func_nb`."""
        if self.flexible_mode:
            if self["post_segment_func_nb"] is None:
                return nb.no_post_func_nb
            return self["post_segment_func_nb"]
        return None

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Argument `staticized`."""
        staticized = self.pre_staticized
        if isinstance(staticized, dict):
            staticized = dict(staticized)
        if self.flexible_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    if isinstance(staticized, dict):
                        adapt_staticized_to_udf(staticized, "ls_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_ls_signal_func_nb"
                elif self.signals_mode:
                    if isinstance(staticized, dict):
                        adapt_staticized_to_udf(staticized, "dir_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_dir_signal_func_nb"
                elif self.order_mode:
                    if isinstance(staticized, dict):
                        adapt_staticized_to_udf(staticized, "order_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_order_signal_func_nb"
            elif isinstance(staticized, dict):
                adapt_staticized_to_udf(staticized, self["signal_func_nb"], "signal_func_nb")
            if self["adjust_func_nb"] is not None and isinstance(staticized, dict):
                adapt_staticized_to_udf(staticized, self["adjust_func_nb"], "adjust_func_nb")
            if self["post_segment_func_nb"] is not None and isinstance(staticized, dict):
                adapt_staticized_to_udf(staticized, self["post_segment_func_nb"], "post_segment_func_nb")
        return staticized

    @cachedproperty
    def pre_adjust_args(self) -> tp.Args:
        """Argument `adjust_args` before template substitution."""
        return self["adjust_args"]

    @cachedproperty
    def pre_signal_args(self) -> tp.Args:
        """Argument `signal_args` before template substitution."""
        return self["signal_args"]

    @cachedproperty
    def pre_post_segment_args(self) -> tp.Args:
        """Argument `post_segment_args` before template substitution."""
        return self["post_segment_args"]

    @cachedproperty
    def pre_chunked(self) -> tp.ChunkedOption:
        """Argument `chunked` before template substitution."""
        return self["chunked"]

    # ############# Ready arguments ############# #

    @cachedproperty
    def save_state(self) -> bool:
        """Argument `save_state`."""
        save_state = self["save_state"]
        if save_state and self.flexible_mode:
            raise ValueError("Argument save_state cannot be used in flexible mode")
        return save_state

    @cachedproperty
    def save_value(self) -> bool:
        """Argument `save_value`."""
        save_value = self["save_value"]
        if save_value and self.flexible_mode:
            raise ValueError("Argument save_value cannot be used in flexible mode")
        return save_value

    @cachedproperty
    def save_returns(self) -> bool:
        """Argument `save_returns`."""
        save_returns = self["save_returns"]
        if save_returns and self.flexible_mode:
            raise ValueError("Argument save_returns cannot be used in flexible mode")
        return save_returns

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre_entries(self) -> tp.ArrayLike:
        """Argument `entries` before broadcasting."""
        return self["entries"] if self["entries"] is not None else False

    @cachedproperty
    def pre_exits(self) -> tp.ArrayLike:
        """Argument `exits` before broadcasting."""
        return self["exits"] if self["exits"] is not None else False

    @cachedproperty
    def pre_short_entries(self) -> tp.ArrayLike:
        """Argument `short_entries` before broadcasting."""
        return self["short_entries"] if self["short_entries"] is not None else False

    @cachedproperty
    def pre_short_exits(self) -> tp.ArrayLike:
        """Argument `short_exits` before broadcasting."""
        return self["short_exits"] if self["short_exits"] is not None else False

    @cachedproperty
    def pre_from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting."""
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def pre_max_logs(self) -> tp.Optional[int]:
        """Argument `max_logs` before broadcasting."""
        return self["max_logs"]

    @cachedproperty
    def pre_in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        if self.flexible_mode:
            return BasePreparer.pre_in_outputs.func(self)
        if self["in_outputs"] is not None:
            raise ValueError("Argument in_outputs cannot be used in fixed mode")
        return None

    # ############# Broadcasting ############# #

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        def_broadcast_kwargs = dict(BasePreparer.def_broadcast_kwargs.func(self))
        if self.order_mode:
            def_broadcast_kwargs["keep_flex"] = dict(
                size=False,
                size_type=False,
                min_size=False,
                max_size=False,
                _def=True,
            )
            def_broadcast_kwargs["min_ndim"] = dict(
                size=2,
                size_type=2,
                min_size=2,
                max_size=2,
                _def=None,
            )
            def_broadcast_kwargs["require_kwargs"] = dict(
                size=dict(requirements="O"),
                size_type=dict(requirements="O"),
                min_size=dict(requirements="O"),
                max_size=dict(requirements="O"),
            )
        if self.stop_ladder:
            def_broadcast_kwargs["axis"] = dict(
                sl_stop=1,
                tsl_stop=1,
                tp_stop=1,
                td_stop=1,
                dt_stop=1,
            )
            def_broadcast_kwargs["merge_kwargs"] = dict(
                sl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tsl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tp_stop=dict(reset_index="from_start", fill_value=np.nan),
                td_stop=dict(reset_index="from_start", fill_value=-1),
                dt_stop=dict(reset_index="from_start", fill_value=-1),
            )
        return def_broadcast_kwargs

    # ############# After broadcasting ############# #

    @cachedproperty
    def signals(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike, tp.ArrayLike, tp.ArrayLike]:
        """Arguments `entries`, `exits`, `short_entries`, and `short_exits` after broadcasting."""
        entries = self.post_entries
        exits = self.post_exits
        short_entries = self.post_short_entries
        short_exits = self.post_short_exits

        if not self.flexible_mode and not self.ls_mode:
            direction = self.post_direction
            if direction.size == 1:
                _direction = direction.item(0)
                if _direction == enums.Direction.LongOnly:
                    long_entries = entries
                    long_exits = exits
                    short_entries = np.array([[False]])
                    short_exits = np.array([[False]])
                elif _direction == enums.Direction.ShortOnly:
                    long_entries = np.array([[False]])
                    long_exits = np.array([[False]])
                    short_entries = entries
                    short_exits = exits
                else:
                    long_entries = entries
                    long_exits = np.array([[False]])
                    short_entries = exits
                    short_exits = np.array([[False]])
            else:
                return nb.dir_to_ls_signals_nb(
                    target_shape=self.target_shape,
                    entries=entries,
                    exits=exits,
                    direction=direction,
                )
        else:
            long_entries, long_exits = entries, exits
        return long_entries, long_exits, short_entries, short_exits

    @cachedproperty
    def entries(self) -> tp.ArrayLike:
        """Argument `entries`."""
        return self.signals[0]

    @cachedproperty
    def exits(self) -> tp.ArrayLike:
        """Argument `exits`."""
        return self.signals[1]

    @cachedproperty
    def short_entries(self) -> tp.ArrayLike:
        """Argument `short_entries`."""
        return self.signals[2]

    @cachedproperty
    def short_exits(self) -> tp.ArrayLike:
        """Argument `short_exits`."""
        return self.signals[3]

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Arguments `price` and `from_ago` after broadcasting."""
        price = self.post_price
        from_ago = self.post_from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                if next_open_mask.any() or next_close_mask.any():
                    price = price.astype(np.float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=np.int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Argument `price`."""
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago`."""
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_logs(self) -> tp.Optional[int]:
        """Argument `max_logs`."""
        max_logs = self.pre_max_logs
        if max_logs is None:
            _log = self.post_log
            if _log.size == 1:
                max_logs = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_logs = self.target_shape[0] * int(np.any(_log))
                else:
                    max_logs = int(np.max(np.sum(_log, axis=0)))
        return max_logs

    @cachedproperty
    def use_stops(self) -> bool:
        """Argument `use_stops`."""
        if self.stop_ladder:
            use_stops = True
        else:
            if self.flexible_mode:
                use_stops = True
            else:
                if (
                    not np.any(self.sl_stop)
                    and not np.any(self.tsl_stop)
                    and not np.any(self.tp_stop)
                    and not np.any(self.td_stop != -1)
                    and not np.any(self.dt_stop != -1)
                ):
                    use_stops = False
                else:
                    use_stops = True
        return use_stops

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                order_mode=self.order_mode,
                use_stops=self.use_stops,
                stop_ladder=self.stop_ladder,
                adjust_func_nb=self.adjust_func_nb,
                adjust_args=self.pre_adjust_args,
                signal_func_nb=self.signal_func_nb,
                signal_args=self.pre_signal_args,
                post_segment_func_nb=self.post_segment_func_nb,
                post_segment_args=self.pre_post_segment_args,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                fill_pos_info=self.fill_pos_info,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_orders=self.max_orders,
                max_logs=self.max_logs,
            ),
            BasePreparer.template_context.func(self),
        )

    @cachedproperty
    def post_adjust_args(self) -> tp.Args:
        """Argument `adjust_args` after template substitution."""
        return substitute_templates(self.pre_adjust_args, self.template_context, sub_id="adjust_args")

    @cachedproperty
    def post_signal_args(self) -> tp.Args:
        """Argument `signal_args` after template substitution."""
        return substitute_templates(self.pre_signal_args, self.template_context, sub_id="signal_args")

    @cachedproperty
    def post_post_segment_args(self) -> tp.Args:
        """Argument `post_segment_args` after template substitution."""
        return substitute_templates(self.pre_post_segment_args, self.template_context, sub_id="post_segment_args")

    @cachedproperty
    def adjust_args(self) -> tp.Args:
        """Argument `adjust_args`."""
        return self.post_adjust_args

    @cachedproperty
    def signal_args(self) -> tp.Args:
        """Argument `signal_args`."""
        if self.flexible_mode:
            if self.ls_mode:
                return (
                    self.entries,
                    self.exits,
                    self.short_entries,
                    self.short_exits,
                    self.from_ago,
                    *((self.adjust_func_nb,) if self.staticized is None else ()),
                    self.adjust_args,
                )
            if self.signals_mode:
                return (
                    self.entries,
                    self.exits,
                    self.direction,
                    self.from_ago,
                    *((self.adjust_func_nb,) if self.staticized is None else ()),
                    self.adjust_args,
                )
            if self.order_mode:
                return (
                    self.size,
                    self.price,
                    self.size_type,
                    self.direction,
                    self.min_size,
                    self.max_size,
                    self.val_price,
                    self.from_ago,
                    *((self.adjust_func_nb,) if self.staticized is None else ()),
                    self.adjust_args,
                )
        return self.post_signal_args

    @cachedproperty
    def post_segment_args(self) -> tp.Args:
        """Argument `post_segment_args`."""
        return self.post_post_segment_args

    @cachedproperty
    def chunked(self) -> tp.ChunkedOption:
        if self.flexible_mode:
            if self.ls_mode:
                return ch.specialize_chunked_option(
                    self.pre_chunked,
                    arg_take_spec=dict(
                        signal_args=ch.ArgsTaker(
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            *((None,) if self.staticized is None else ()),
                            ch.ArgsTaker(),
                        )
                    ),
                )
            if self.signals_mode:
                return ch.specialize_chunked_option(
                    self.pre_chunked,
                    arg_take_spec=dict(
                        signal_args=ch.ArgsTaker(
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            *((None,) if self.staticized is None else ()),
                            ch.ArgsTaker(),
                        )
                    ),
                )
            if self.order_mode:
                return ch.specialize_chunked_option(
                    self.pre_chunked,
                    arg_take_spec=dict(
                        signal_args=ch.ArgsTaker(
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            base_ch.flex_array_gl_slicer,
                            *((None,) if self.staticized is None else ()),
                            ch.ArgsTaker(),
                        )
                    ),
                )
        return self.pre_chunked

    # ############# Result ############# #

    @cachedproperty
    def sim_func(self) -> tp.Optional[tp.Callable]:
        if self.flexible_mode:
            func = resolve_dynamic_simulator("from_signal_func_nb", self.staticized)
            func = jit_reg.resolve_option(func, self.jitted)
            func = ch_reg.resolve_option(func, self.chunked)
        else:
            func = jit_reg.resolve_option(nb.from_signals_nb, self.jitted)
            func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def sim_arg_map(self) -> tp.Kwargs:
        sim_arg_map = {}
        if self.flexible_mode:
            if self.staticized is not None:
                sim_arg_map["signal_func_nb"] = None
                sim_arg_map["post_segment_func_nb"] = None
        else:
            sim_arg_map["long_entries"] = "entries"
            sim_arg_map["long_exits"] = "exits"
            sim_arg_map["group_lens"] = "cs_group_lens"
        return sim_arg_map

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        pf_args = dict(BasePreparer.pf_args.func(self))
        pf_args["orders_cls"] = FSOrders
        return pf_args


FSPreparer.override_arg_config_doc(__pdoc__)
