# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for preparing portfolio simulations."""

import inspect
import string
import warnings
import attr
from collections import defaultdict
from functools import cached_property as cachedproperty

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, ReadonlyConfig, HybridConfig, Configured, merge_dicts
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.params import Param
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.base.reshaping import BCO, Default, Ref, broadcast, broadcast_array_to
from vectorbtpro.base.indexing import index_dict
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import Data
from vectorbtpro.portfolio import enums
from vectorbtpro.portfolio.call_seq import require_call_seq, build_call_seq
from vectorbtpro.portfolio.decorators import override_arg_config, attach_arg_properties

__all__ = [
    "BasePreparer",
    "FOPreparer",
]

__pdoc__ = {}


base_arg_config = ReadonlyConfig(
    dict(
        args=dict(
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
            attach_call_seq=dict(),
            freq=dict(),
            broadcast_kwargs=dict(is_dict=True),
            template_context=dict(is_dict=True),
            seed=dict(),
        ),
        broadcast_kwargs=dict(
            to_pd=False,
            keep_flex=True,
            wrapper_kwargs=dict(
                freq=RepEval("self.freq"),
                group_by=RepEval("self.group_by"),
            ),
            return_wrapper=True,
        ),
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
        arg_config = self.arg_config["args"][arg_name]
        if "map_enum_kwargs" in arg_config:
            return self.get_mapped_arg_default(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                **arg_config["map_enum_kwargs"],
            )
        return self.get_raw_arg_default(
            arg_name,
            is_dict=arg_config.get("is_dict", False),
        )

    def get_arg(self, arg_name: str) -> tp.Any:
        """Get mapped argument according to the argument config."""
        arg_config = self.arg_config["args"][arg_name]
        if "map_enum_kwargs" in arg_config:
            return self.get_mapped_arg(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                has_default=arg_config.get("has_default", True),
                **arg_config["map_enum_kwargs"],
            )
        return self.get_raw_arg(
            arg_name,
            is_dict=arg_config.get("is_dict", False),
            has_default=arg_config.get("has_default", True),
        )

    def prepare_post_arg(self, arg_name: str) -> tp.ArrayLike:
        """Prepare an argument after broadcasting."""
        arg = self.post_args[arg_name]
        if arg is not None:
            arg_config = self.arg_config["args"][arg_name]
            if "map_enum_kwargs" in arg_config:
                arg = map_enum_fields(arg, **arg_config["map_enum_kwargs"])
            if "subdtype" in arg_config:
                checks.assert_subdtype(arg, arg_config["subdtype"], arg_name=arg_name)
        return arg

    # ############# Before broadcasting ############# #

    @cachedproperty
    def data(self) -> tp.Optional[Data]:
        """Data instance."""
        return self.get_arg("data")

    @cachedproperty
    def pre_open(self) -> tp.ArrayLike:
        """Argument `open` before broadcasting."""
        open = self.get_arg("open")
        if open is None:
            if self.data is not None:
                open = self.data.open
            if open is None:
                return np.nan
        return open

    @cachedproperty
    def open_none(self) -> bool:
        """Whether argument `open` is None."""
        return self.pre_open is np.nan

    @cachedproperty
    def pre_high(self) -> tp.ArrayLike:
        """Argument `high` before broadcasting."""
        high = self.get_arg("high")
        if high is None:
            if self.data is not None:
                high = self.data.high
            if high is None:
                return np.nan
        return high

    @cachedproperty
    def high_none(self) -> bool:
        """Whether argument `high` is None."""
        return self.pre_high is np.nan

    @cachedproperty
    def pre_low(self) -> tp.ArrayLike:
        """Argument `low` before broadcasting."""
        low = self.get_arg("low")
        if low is None:
            if self.data is not None:
                low = self.data.low
            if low is None:
                return np.nan
        return low

    @cachedproperty
    def low_none(self) -> bool:
        """Whether argument `low` is None."""
        return self.pre_low is np.nan

    @cachedproperty
    def pre_close(self) -> tp.ArrayLike:
        """Argument `close` before broadcasting."""
        close = self.get_arg("close")
        if close is None:
            if self.data is not None:
                close = self.data.close
            if close is None:
                return np.nan
        return close

    @cachedproperty
    def pre_bm_close(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `bm_close` before broadcasting."""
        bm_close = self.get_arg("bm_close")
        if bm_close is not None and not isinstance(bm_close, bool):
            return bm_close
        return None

    @cachedproperty
    def init_cash_mode(self) -> tp.Optional[int]:
        """Initial cash mode."""
        init_cash = self.get_arg("init_cash")
        if init_cash in enums.InitCashMode:
            return init_cash
        return None

    @cachedproperty
    def pre_init_cash(self) -> tp.ArrayLike:
        """Argument `init_cash` before broadcasting."""
        if self.init_cash_mode is not None:
            return np.inf
        return self.get_arg("init_cash")

    @cachedproperty
    def pre_init_position(self) -> tp.ArrayLike:
        """Argument `init_position` before broadcasting."""
        return self.get_arg("init_position")

    @cachedproperty
    def pre_init_price(self) -> tp.ArrayLike:
        """Argument `init_price` before broadcasting."""
        return self.get_arg("init_price")

    @cachedproperty
    def pre_cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits` before broadcasting."""
        return self.get_arg("cash_deposits")

    @cachedproperty
    def pre_cash_earnings(self) -> tp.ArrayLike:
        """Argument `cash_earnings` before broadcasting."""
        return self.get_arg("cash_earnings")

    @cachedproperty
    def cash_sharing(self) -> bool:
        """Argument `cash_sharing`."""
        return self.get_arg("cash_sharing")

    @cachedproperty
    def freq(self) -> tp.Optional[tp.FrequencyLike]:
        """Argument `freq`."""
        freq = self.get_arg("freq")
        if freq is None and self.data is not None:
            return self.data.freq
        return freq

    @cachedproperty
    def group_by(self) -> tp.GroupByLike:
        """Argument `group_by`."""
        group_by = self.get_arg("group_by")
        if group_by is None and self.cash_sharing:
            return True
        return group_by

    @cachedproperty
    def attach_call_seq(self) -> bool:
        """Argument `attach_call_seq`."""
        return self.get_arg("attach_call_seq")

    @cachedproperty
    def seed(self) -> tp.Optional[int]:
        """Argument `seed`."""
        return self.get_arg("seed")

    def set_seed(self) -> None:
        """Set seed."""
        seed = self.seed
        if seed is not None:
            set_seed(seed)

    # ############# Broadcasting ############# #

    @cachedproperty
    def pre_args(self) -> tp.Kwargs:
        """Arguments before broadcasting."""
        pre_args = dict()
        for k, v in self.arg_config["args"].items():
            if v.get("broadcast", False):
                pre_args[k] = getattr(self, "pre_" + k)
        return pre_args

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        """Argument `template_context`."""
        return merge_dicts(dict(self=self), self.get_arg("template_context"))

    @cachedproperty
    def broadcast_kwargs(self) -> tp.Kwargs:
        """Argument `broadcast_kwargs`."""
        arg_broadcast_kwargs = defaultdict(dict)
        for k, v in self.arg_config["args"].items():
            if v.get("broadcast", False):
                broadcast_kwargs = v.get("broadcast_kwargs", None)
                if broadcast_kwargs is None:
                    broadcast_kwargs = {}
                for k2, v2 in broadcast_kwargs.items():
                    arg_broadcast_kwargs[k2][k] = v2
        broadcast_kwargs = merge_dicts(
            self.arg_config["broadcast_kwargs"],
            dict(arg_broadcast_kwargs),
            self.get_arg("broadcast_kwargs"),
        )
        return substitute_templates(
            broadcast_kwargs,
            self.template_context,
            sub_id="broadcast_kwargs",
        )

    @cachedproperty
    def broadcast_result(self) -> tp.Any:
        """Result of broadcasting."""
        return broadcast(self.pre_args, **self.broadcast_kwargs)

    @cachedproperty
    def post_args(self) -> tp.Kwargs:
        """Arguments after broadcasting."""
        return self.broadcast_result[0]

    @cachedproperty
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self.broadcast_result[1]

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
        cash_deposits = self.get_arg("cash_deposits")
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
    def cash_earnings(self) -> tp.ArrayLike:
        """Argument `cash_earnings`."""
        cash_earnings = self.post_args["cash_earnings"]
        checks.assert_subdtype(cash_earnings, np.number, arg_name="cash_earnings")
        return cash_earnings

    # ############# Preparation ############# #

    @cachedproperty
    def sim_args(self) -> tp.Kwargs:
        """Arguments to be passed to the simulation."""
        return dict(
            target_shape=self.target_shape,
            group_lens=self.group_lens,
            cash_sharing=self.cash_sharing,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            index=self.wrapper.ns_index,
            freq=self.wrapper.ns_freq,
            init_cash=self.init_cash,
            init_position=self.init_position,
            init_price=self.init_price,
            cash_deposits=self.cash_deposits,
            cash_earnings=self.cash_earnings,
        )

    @cachedproperty
    def pf_args(self) -> tp.Kwargs:
        """Arguments to be passed to the portfolio."""
        kwargs = dict()
        for k, v in self.config.items():
            if k not in self.arg_config["args"]:
                kwargs[k] = v
        return dict(
            wrapper=self.wrapper,
            open=self.open if not self.open_none else None,
            high=self.high if not self.high_none else None,
            low=self.low if not self.low_none else None,
            close=self.close,
            cash_sharing=self.cash_sharing,
            init_cash=self.init_cash if self.init_cash_mode is None else self.init_cash_mode,
            init_position=self.init_position,
            init_price=self.init_price,
            bm_close=self.bm_close,
            **kwargs,
        )

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
        args=dict(
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
            call_seq=dict(map_enum_kwargs=dict(enum=enums.CallSeqType, look_for_type=str)),
            ffill_val_price=dict(),
            update_value=dict(),
            save_state=dict(),
            save_value=dict(),
            save_returns=dict(),
            max_orders=dict(),
            max_logs=dict(),
        ),
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
    def from_ago_none(self) -> bool:
        """Whether the initial value of argument `from_ago` is None."""
        return self.get_arg("from_ago") is None

    @cachedproperty
    def pre_from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting."""
        from_ago = self.get_arg("from_ago")
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def auto_call_seq(self) -> bool:
        """Whether automatic call sequence is enabled."""
        call_seq = self.get_arg("call_seq")
        return checks.is_int(call_seq) and call_seq == enums.CallSeqType.Auto

    @cachedproperty
    def pre_call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq` before broadcasting."""
        if self.auto_call_seq:
            return None
        return self.get_arg("call_seq")

    @cachedproperty
    def ffill_val_price(self) -> bool:
        """Argument `ffill_val_price`."""
        return self.get_arg("ffill_val_price")

    @cachedproperty
    def update_value(self) -> bool:
        """Argument `update_value`."""
        return self.get_arg("update_value")

    @cachedproperty
    def save_state(self) -> bool:
        """Argument `save_state`."""
        return self.get_arg("save_state")

    @cachedproperty
    def save_value(self) -> bool:
        """Argument `save_value`."""
        return self.get_arg("save_value")

    @cachedproperty
    def save_returns(self) -> bool:
        """Argument `save_returns`."""
        return self.get_arg("save_returns")

    @cachedproperty
    def pre_max_orders(self) -> tp.Optional[int]:
        """Argument `max_orders` before broadcasting."""
        return self.get_arg("max_orders")

    @cachedproperty
    def pre_max_logs(self) -> tp.Optional[int]:
        """Argument `max_logs` before broadcasting."""
        return self.get_arg("max_logs")

    # ############# After broadcasting ############# #

    @cachedproperty
    def extra_post_args(self) -> tp.Kwargs:
        """Arguments after extra preparation."""
        price = self.post_price
        from_ago = self.post_from_ago
        if self.from_ago_none:
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
        return dict(price=price, from_ago=from_ago)

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Argument `price`."""
        return self.extra_post_args["price"]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago`."""
        return self.extra_post_args["from_ago"]

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

    @cachedproperty
    def max_orders(self) -> tp.Optional[int]:
        """Argument `max_orders`."""
        max_orders = self.pre_max_orders
        if max_orders is None:
            _size = self.size
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
            _log = self.post_args["log"]
            if _log.size == 1:
                max_logs = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_logs = self.target_shape[0] * int(np.any(_log))
                else:
                    max_logs = int(np.max(np.sum(_log, axis=0)))
        return max_logs

    @cachedproperty
    def sim_args(self) -> tp.Kwargs:
        """Arguments to be passed to the simulation."""
        return dict(
            target_shape=self.target_shape,
            group_lens=self.cs_group_lens,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            init_cash=self.init_cash,
            init_position=self.init_position,
            init_price=self.init_price,
            cash_deposits=self.cash_deposits,
            cash_earnings=self.cash_earnings,
            cash_dividends=self.cash_dividends,
            size=self.size,
            price=self.price,
            size_type=self.size_type,
            direction=self.direction,
            fees=self.fees,
            fixed_fees=self.fixed_fees,
            slippage=self.slippage,
            min_size=self.min_size,
            max_size=self.max_size,
            size_granularity=self.size_granularity,
            leverage=self.leverage,
            leverage_mode=self.leverage_mode,
            reject_prob=self.reject_prob,
            price_area_vio_mode=self.price_area_vio_mode,
            allow_partial=self.allow_partial,
            raise_reject=self.raise_reject,
            log=self.log,
            val_price=self.val_price,
            from_ago=self.from_ago,
            call_seq=self.call_seq,
            auto_call_seq=self.auto_call_seq,
            ffill_val_price=self.ffill_val_price,
            update_value=self.update_value,
            save_state=self.save_state,
            save_value=self.save_value,
            save_returns=self.save_returns,
            max_orders=self.max_orders,
            max_logs=self.max_logs,
        )


FOPreparer.override_arg_config_doc(__pdoc__)
