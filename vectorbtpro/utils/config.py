# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for configuration."""

import warnings
import inspect
from collections import namedtuple
from copy import copy, deepcopy

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.caching import Cacheable
from vectorbtpro.utils.decorators import class_or_instancemethod
from vectorbtpro.utils.formatting import Prettified, prettify_dict, prettify_inited
from vectorbtpro.utils.pickling import Pickleable


def resolve_dict(dct: tp.DictLikeSequence, i: tp.Optional[int] = None) -> dict:
    """Select keyword arguments."""
    if dct is None:
        dct = {}
    if isinstance(dct, dict):
        return dict(dct)
    if i is not None:
        _dct = dct[i]
        if _dct is None:
            _dct = {}
        return dict(_dct)
    raise ValueError("Cannot resolve dict")


class atomic_dict(dict):
    """Dict that behaves like a single value when merging."""

    pass


InConfigLikeT = tp.Union[None, dict, "ConfigT"]
OutConfigLikeT = tp.Union[dict, "ConfigT"]


def convert_to_dict(dct: InConfigLikeT, nested: bool = True) -> dict:
    """Convert any config to `dict`.

    Set `nested` to True to convert all child dicts in recursive manner."""
    if dct is None:
        dct = {}
    if isinstance(dct, Config):
        dct = dict(dct)
    else:
        dct = type(dct)(dct)
    if not nested:
        return dct
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = convert_to_dict(v, nested=nested)
        else:
            dct[k] = v
    return dct


def get_dict_item(dct: dict, k: tp.Hashable) -> tp.Any:
    """Get dict item under the key `k`.

    The key can be nested using the dot notation or tuple, and must be hashable."""
    if k in dct:
        return dct[k]
    if isinstance(k, str) and '.' in k:
        k = tuple(k.split('.'))
    if isinstance(k, tuple):
        if len(k) == 1:
            return dct[k[0]]
        return get_dict_item(dct[k[0]], k[1:])
    return dct[k]


def set_dict_item(dct: dict, k: tp.Any, v: tp.Any, force: bool = False) -> None:
    """Set dict item.

    If the dict is of the type `Config`, also passes `force` keyword to override blocking flags."""
    if isinstance(dct, Config):
        dct.__setitem__(k, v, force=force)
    else:
        dct[k] = v


def copy_dict(dct: InConfigLikeT, copy_mode: str = "shallow", nested: bool = True) -> OutConfigLikeT:
    """Copy dict based on a copy mode.

    The following modes are supported:

    * 'none': Does not copy
    * 'shallow': Copies keys only
    * 'hybrid': Copies keys and values using `copy.copy`
    * 'deep': Copies the whole thing using `copy.deepcopy`

    Set `nested` to True to copy all child dicts in recursive manner."""
    if dct is None:
        return {}
    copy_mode = copy_mode.lower()
    if copy_mode not in {"none", "shallow", "hybrid", "deep"}:
        raise ValueError(f"Copy mode '{copy_mode}' not supported")

    if copy_mode == "none":
        return dct
    if copy_mode == "deep":
        return deepcopy(dct)
    if isinstance(dct, Config):
        return dct.copy(copy_mode=copy_mode, nested=nested)
    dct_copy = copy(dct)  # copy structure using shallow copy
    for k, v in dct_copy.items():
        if nested and isinstance(v, dict):
            _v = copy_dict(v, copy_mode=copy_mode, nested=nested)
        else:
            if copy_mode == "hybrid":
                _v = copy(v)  # copy values using shallow copy
            else:
                _v = v
        set_dict_item(dct_copy, k, _v, force=True)
    return dct_copy


def update_dict(
    x: InConfigLikeT,
    y: InConfigLikeT,
    nested: bool = True,
    force: bool = False,
    same_keys: bool = False,
) -> None:
    """Update dict with keys and values from other dict.

    Set `nested` to True to update all child dicts in recursive manner.
    For `force`, see `set_dict_item`.

    If you want to treat any dict as a single value, wrap it with `atomic_dict`.

    !!! note
        If the child dict is not atomic, it will copy only its values, not its meta."""
    if x is None:
        return
    if y is None:
        return
    checks.assert_instance_of(x, dict)
    checks.assert_instance_of(y, dict)

    for k, v in y.items():
        if nested and k in x and isinstance(x[k], dict) and isinstance(v, dict) and not isinstance(v, atomic_dict):
            update_dict(x[k], v, force=force)
        else:
            if same_keys and k not in x:
                continue
            set_dict_item(x, k, v, force=force)


def merge_dicts(
    *dicts: InConfigLikeT,
    to_dict: bool = True,
    copy_mode: str = "shallow",
    nested: tp.Optional[bool] = None,
    same_keys: bool = False,
) -> OutConfigLikeT:
    """Merge dicts.

    Args:
        *dicts (dict): Dicts.
        to_dict (bool): Whether to call `convert_to_dict` on each dict prior to copying.
        copy_mode (str): Mode for `copy_dict` to copy each dict prior to merging.
        nested (bool): Whether to merge all child dicts in recursive manner.

            If None, checks whether any dict is nested.
        same_keys (bool): Whether to merge on the overlapping keys only."""
    # Shortcut when both dicts are None
    if dicts[0] is None and dicts[1] is None:
        if len(dicts) > 2:
            return merge_dicts(
                None,
                *dicts[2:],
                to_dict=to_dict,
                copy_mode=copy_mode,
                nested=nested,
                same_keys=same_keys,
            )
        return {}

    # Check whether any dict is nested
    if nested is None:
        for dct in dicts:
            if dct is not None:
                for v in dct.values():
                    if isinstance(v, dict) and not isinstance(v, atomic_dict):
                        nested = True
                        break
            if nested:
                break

    # Convert dict-like objects to regular dicts
    if to_dict:
        # Shortcut when all dicts are already regular
        if not nested and copy_mode in {"none", "shallow"}:  # shortcut
            out = {}
            for dct in dicts:
                if dct is not None:
                    out.update(dct)
            return out
        dicts = tuple([convert_to_dict(dct, nested=True) for dct in dicts])

    # Copy all dicts
    if not to_dict or copy_mode not in {"none", "shallow"}:
        # to_dict already does a shallow copy
        dicts = tuple([copy_dict(dct, copy_mode=copy_mode, nested=nested) for dct in dicts])

    # Merge both dicts
    x, y = dicts[0], dicts[1]
    should_update = True
    if type(x) is dict and type(y) is dict and len(x) == 0:
        x = y
        should_update = False
    if isinstance(x, atomic_dict) or isinstance(y, atomic_dict):
        x = y
        should_update = False
    if should_update:
        update_dict(x, y, nested=nested, force=True, same_keys=same_keys)

    # Merge resulting dict with remaining dicts
    if len(dicts) > 2:
        return merge_dicts(
            x,
            *dicts[2:],
            to_dict=False,  # executed only once
            copy_mode="none",  # executed only once
            nested=nested,
            same_keys=same_keys,
        )
    return x


_RaiseKeyError = object()

DumpTuple = namedtuple("DumpTuple", ("cls", "dumps"))

PickleableDictT = tp.TypeVar("PickleableDictT", bound="PickleableDict")


class PickleableDict(Pickleable, dict):
    """Dict that may contain values of type `Pickleable`."""

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        dct = {}
        for k, v in self.items():
            if isinstance(v, Pickleable):
                dct[k] = DumpTuple(cls=type(v), dumps=v.dumps(**kwargs))
            else:
                dct[k] = v
        return pickle.dumps(dct, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableDictT], dumps: bytes, **kwargs) -> PickleableDictT:
        """Unpickle from bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        config = pickle.loads(dumps, **kwargs)
        for k, v in config.items():
            if isinstance(v, DumpTuple):
                config[k] = v.cls.loads(v.dumps, **kwargs)
        return cls(**config)

    def load_update(self, path: tp.Optional[tp.PathLike] = None, clear: bool = False, **kwargs) -> None:
        """Load dumps from a file and update this instance in-place."""
        if clear:
            self.clear()
        self.update(self.load(path=path, **kwargs))


class ChildDict(dict):
    """Subclass of `dict` acting as a child dict."""

    pass


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(PickleableDict, Prettified):
    """Extends pickleable dict with config features such as nested updates, freezing, and resetting.

    Args:
        *args: Arguments to construct the dict from.
        copy_kwargs_ (dict): Keyword arguments passed to `copy_dict` for copying main dict and `reset_dct_`.

            Copy mode defaults to 'none'.
        reset_dct_ (dict): Dict to fall back to in case of resetting.

            Defaults to None. If None, copies main dict using `reset_dct_copy_kwargs_`.

            !!! note
                Defaults to main dict in case it's None and `readonly_` is True.
        reset_dct_copy_kwargs_ (dict): Keyword arguments that override `copy_kwargs_` for `reset_dct_`.

            Copy mode defaults to 'none' if `readonly_` is True, else to 'hybrid'.
        frozen_keys_ (bool): Whether to deny updates to the keys of the config.

            Defaults to False.
        readonly_ (bool): Whether to deny updates to the keys and values of the config.

            Defaults to False.
        nested_ (bool): Whether to do operations recursively on each child dict.

            Such operations include copy, update, and merge.
            Disable to treat each child dict as a single value. Defaults to True.
        convert_children_ (bool or type): Whether to convert child dicts of type `ChildDict` to configs
            with the same configuration.

            This will trigger a waterfall reaction across all child dicts. Won't convert dicts that
            are already configs. Apart from boolean, you can set it to any subclass of `Config` to use
            it for construction. Requires `nested_` to be True. Defaults to False.
        as_attrs_ (bool): Whether to enable accessing dict keys via the dot notation.

            Enables autocompletion (but only during runtime!). Raises error in case of naming conflicts.
            Defaults to True if `frozen_keys_` or `readonly_`, otherwise False.

            To make nested dictionaries also accessible via the dot notation, wrap
            them with `ChildDict` and set `convert_children_` and `nested_` to True.
        **kwargs: Keyword arguments to construct the dict from.

    Defaults can be overridden with settings under `vectorbtpro._settings.config`.

    If another config is passed, its properties are copied over, but they can still be overridden
    with the arguments passed to the initializer.

    !!! note
        All arguments are applied only once during initialization.
    """

    def __init__(
        self,
        *args,
        copy_kwargs_: tp.KwargsLike = None,
        reset_dct_: tp.DictLike = None,
        reset_dct_copy_kwargs_: tp.KwargsLike = None,
        frozen_keys_: tp.Optional[bool] = None,
        readonly_: tp.Optional[bool] = None,
        nested_: tp.Optional[bool] = None,
        convert_children_: tp.Optional[tp.Union[bool, tp.Type["Config"]]] = None,
        as_attrs_: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        try:
            from vectorbtpro._settings import settings

            config_cfg = settings["config"]
        except ImportError:
            config_cfg = {}

        # Build dict
        if len(args) > 0 and isinstance(args[0], Config):
            cfg = args[0]
        else:
            cfg = None
        dct = dict(*args, **kwargs)

        # Resolve params
        def _resolve_param(pname: str, p: tp.Any, default: tp.Any, merge: bool = False) -> tp.Any:
            cfg_default = config_cfg.get(pname, None)
            if cfg is None:
                dct_p = None
            else:
                dct_p = getattr(cfg, pname)

            if merge and isinstance(default, dict):
                return merge_dicts(default, cfg_default, dct_p, p)
            if p is not None:
                return p
            if dct_p is not None:
                return dct_p
            if cfg_default is not None:
                return cfg_default
            return default

        reset_dct_ = _resolve_param("reset_dct_", reset_dct_, None)
        frozen_keys_ = _resolve_param("frozen_keys_", frozen_keys_, False)
        readonly_ = _resolve_param("readonly_", readonly_, False)
        nested_ = _resolve_param("nested_", nested_, True)
        convert_children_ = _resolve_param("convert_children_", convert_children_, False)
        as_attrs_ = _resolve_param("as_attrs_", as_attrs_, False)
        reset_dct_copy_kwargs_ = merge_dicts(copy_kwargs_, reset_dct_copy_kwargs_)
        copy_kwargs_ = _resolve_param("copy_kwargs_", copy_kwargs_, dict(copy_mode="none", nested=nested_), merge=True)
        reset_dct_copy_kwargs_ = _resolve_param(
            "reset_dct_copy_kwargs_",
            reset_dct_copy_kwargs_,
            dict(copy_mode="none" if readonly_ else "hybrid", nested=nested_),
            merge=True,
        )

        # Copy dict
        dct = copy_dict(dict(dct), **copy_kwargs_)

        # Convert child dicts
        if convert_children_ and nested_:
            for k, v in dct.items():
                if isinstance(v, ChildDict):
                    if isinstance(convert_children_, bool):
                        config_cls = type(self)
                    elif issubclass(convert_children_, Config):
                        config_cls = convert_children_
                    else:
                        raise TypeError("convert_children_ must be either boolean or a subclass of Config")
                    dct[k] = config_cls(
                        v,
                        copy_kwargs_=copy_kwargs_,
                        reset_dct_copy_kwargs_=reset_dct_copy_kwargs_,
                        frozen_keys_=frozen_keys_,
                        readonly_=readonly_,
                        nested_=nested_,
                        convert_children_=convert_children_,
                        as_attrs_=as_attrs_,
                    )

        # Copy initial config
        if reset_dct_ is None:
            reset_dct_ = dct
        reset_dct_ = copy_dict(dict(reset_dct_), **reset_dct_copy_kwargs_)

        dict.__init__(self, dct)

        self._copy_kwargs_ = copy_kwargs_
        self._reset_dct_ = reset_dct_
        self._reset_dct_copy_kwargs_ = reset_dct_copy_kwargs_
        self._frozen_keys_ = frozen_keys_
        self._readonly_ = readonly_
        self._nested_ = nested_
        self._convert_children_ = convert_children_
        self._as_attrs_ = as_attrs_

        # Set keys as attributes for autocomplete
        if as_attrs_:
            self_dir = set(self.__dir__())
            for k, v in self.items():
                if k in self_dir:
                    raise ValueError(f"Key '{k}' shadows an attribute of the config. Disable as_attrs_.")

    @property
    def copy_kwargs_(self) -> tp.Kwargs:
        """Parameters for copying main dict."""
        return self._copy_kwargs_

    @property
    def reset_dct_(self) -> dict:
        """Dict to fall back to in case of resetting."""
        return self._reset_dct_

    @property
    def reset_dct_copy_kwargs_(self) -> tp.Kwargs:
        """Parameters for copying `reset_dct_`."""
        return self._reset_dct_copy_kwargs_

    @property
    def frozen_keys_(self) -> bool:
        """Whether to deny updates to the keys and values of the config."""
        return self._frozen_keys_

    @property
    def readonly_(self) -> bool:
        """Whether to deny any updates to the config."""
        return self._readonly_

    @property
    def nested_(self) -> bool:
        """Whether to do operations recursively on each child dict."""
        return self._nested_

    @property
    def convert_children_(self) -> tp.Union[bool, tp.Type["Config"]]:
        """Whether to convert child dicts of type `ChildDict` to configs with the same configuration."""
        return self._convert_children_

    @property
    def as_attrs_(self) -> bool:
        """Whether to enable accessing dict keys via dot notation."""
        return self._as_attrs_

    def _safe_getattr(self, k: str, v: tp.Any) -> bool:
        """Get an attribute without triggering `Config.__getattr__`."""
        try:
            return object.__getattribute__(self, k)
        except AttributeError:
            return v

    def __getattr__(self, k: str) -> tp.Any:
        try:
            _as_attrs_ = object.__getattribute__(self, "_as_attrs_")
        except AttributeError:
            return object.__getattribute__(self, k)
        if _as_attrs_:
            try:
                return self.__getitem__(k)
            except KeyError:
                raise AttributeError
        return object.__getattribute__(self, k)

    def __setattr__(self, k: str, v: tp.Any, force: bool = False) -> None:
        try:
            _as_attrs_ = object.__getattribute__(self, "_as_attrs_")
        except AttributeError:
            return object.__setattr__(self, k, v)
        if _as_attrs_:
            return self.__setitem__(k, v, force=force)
        return object.__setattr__(self, k, v)

    def __delattr__(self, k: str, force: bool = False) -> None:
        try:
            _as_attrs_ = object.__getattribute__(self, "_as_attrs_")
        except AttributeError:
            return object.__delattr__(self, k)
        if _as_attrs_:
            return self.__delitem__(k, force=force)
        return object.__delattr__(self, k)

    def __setitem__(self, k: str, v: tp.Any, force: bool = False) -> None:
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        if not force and self._safe_getattr("frozen_keys_", False):
            if k not in self:
                raise KeyError(f"Config keys are frozen: key '{k}' not found")
        dict.__setitem__(self, k, v)

    def __delitem__(self, k: str, force: bool = False) -> None:
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        if not force and self._safe_getattr("frozen_keys_", False):
            raise KeyError(f"Config keys are frozen")
        dict.__delitem__(self, k)

    def pop(self, k: str, v: tp.Any = _RaiseKeyError, force: bool = False) -> tp.Any:
        """Remove and return the pair by the key."""
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        if not force and self._safe_getattr("frozen_keys_", False):
            raise KeyError(f"Config keys are frozen")
        if v is _RaiseKeyError:
            result = dict.pop(self, k)
        else:
            result = dict.pop(self, k, v)
        return result

    def popitem(self, force: bool = False) -> tp.Tuple[tp.Any, tp.Any]:
        """Remove and return some pair."""
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        if not force and self._safe_getattr("frozen_keys_", False):
            raise KeyError(f"Config keys are frozen")
        result = dict.popitem(self)
        return result

    def clear(self, force: bool = False) -> None:
        """Remove all items."""
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        if not force and self._safe_getattr("frozen_keys_", False):
            raise KeyError(f"Config keys are frozen")
        dict.clear(self)

    def update(self, *args, nested: tp.Optional[bool] = None, force: bool = False, **kwargs) -> None:
        """Update the config.

        See `update_dict`."""
        other = dict(*args, **kwargs)
        if nested is None:
            nested = self.nested_
        update_dict(self, other, nested=nested, force=force)

    def __copy__(self: ConfigT) -> ConfigT:
        """Shallow operation, primarily used by `copy.copy`.

        Does not take into account copy parameters."""
        cls = type(self)
        self_copy = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = v
        self_copy.clear(force=True)
        self_copy.update(copy(dict(self)), nested=False, force=True)
        return self_copy

    def __deepcopy__(self: ConfigT, memo: tp.DictLike = None) -> ConfigT:
        """Deep operation, primarily used by `copy.deepcopy`.

        Does not take into account copy parameters."""
        if memo is None:
            memo = {}
        cls = type(self)
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = deepcopy(v, memo)
        self_copy.clear(force=True)
        self_copy.update(deepcopy(dict(self), memo), nested=False, force=True)
        return self_copy

    def copy(
        self: ConfigT,
        reset_dct_copy_kwargs: tp.KwargsLike = None,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
    ) -> ConfigT:
        """Copy the instance.

        By default, copies in the same way as during the initialization."""
        if copy_mode is None:
            copy_mode = self.copy_kwargs_["copy_mode"]
            reset_dct_copy_mode = self.reset_dct_copy_kwargs_["copy_mode"]
        else:
            reset_dct_copy_mode = copy_mode
        if nested is None:
            nested = self.copy_kwargs_["nested"]
            reset_dct_nested = self.reset_dct_copy_kwargs_["nested"]
        else:
            reset_dct_nested = nested
        reset_dct_copy_kwargs = resolve_dict(reset_dct_copy_kwargs)
        if "copy_mode" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["copy_mode"] is not None:
                reset_dct_copy_mode = reset_dct_copy_kwargs["copy_mode"]
        if "nested" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["nested"] is not None:
                reset_dct_nested = reset_dct_copy_kwargs["nested"]

        self_copy = self.__copy__()

        reset_dct_ = copy_dict(dict(self.reset_dct_), copy_mode=reset_dct_copy_mode, nested=reset_dct_nested)
        self.__dict__["_reset_dct_"] = reset_dct_

        dct = copy_dict(dict(self), copy_mode=copy_mode, nested=nested)
        self_copy.update(dct, nested=False, force=True)

        return self_copy

    def merge_with(
        self: ConfigT,
        other: InConfigLikeT,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> OutConfigLikeT:
        """Merge with another dict into one single dict.

        See `merge_dicts`."""
        if copy_mode is None:
            copy_mode = "shallow"
        if nested is None:
            nested = self.nested_
        return merge_dicts(self, other, copy_mode=copy_mode, nested=nested, **kwargs)

    def to_dict(self, nested: tp.Optional[bool] = None) -> dict:
        """Convert to dict."""
        return convert_to_dict(self, nested=nested)

    def reset(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Clears the config and updates it with the initial config.

        `reset_dct_copy_kwargs` override `Config.reset_dct_copy_kwargs_`."""
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct_ = copy_dict(dict(self.reset_dct_), **reset_dct_copy_kwargs)
        self.clear(force=True)
        self.update(self.reset_dct_, nested=False, force=True)
        self.__dict__["_reset_dct_"] = reset_dct_

    def make_checkpoint(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Replace `reset_dct_` by the current state.

        `reset_dct_copy_kwargs` override `Config.reset_dct_copy_kwargs_`."""
        if not force and self._safe_getattr("readonly_", False):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct_ = copy_dict(dict(self), **reset_dct_copy_kwargs)
        self.__dict__["_reset_dct_"] = reset_dct_

    def dumps(self, dump_reset_dct: bool = False, **kwargs) -> bytes:
        """Pickle to bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        if dump_reset_dct:
            reset_dct_ = PickleableDict(self.reset_dct_).dumps(**kwargs)
        else:
            reset_dct_ = None
        return pickle.dumps(
            dict(
                dct=PickleableDict(self).dumps(**kwargs),
                copy_kwargs_=self.copy_kwargs_,
                reset_dct_=reset_dct_,
                reset_dct_copy_kwargs_=self.reset_dct_copy_kwargs_,
                frozen_keys_=self.frozen_keys_,
                readonly_=self.readonly_,
                nested_=self.nested_,
                convert_children_=self.convert_children_,
                as_attrs_=self.as_attrs_,
            ),
            **kwargs,
        )

    @classmethod
    def loads(cls: tp.Type[ConfigT], dumps: bytes, **kwargs) -> ConfigT:
        """Unpickle from bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        obj = pickle.loads(dumps, **kwargs)
        if obj["reset_dct_"] is not None:
            reset_dct_ = PickleableDict.loads(obj["reset_dct_"], **kwargs)
        else:
            reset_dct_ = None
        return cls(
            PickleableDict.loads(obj["dct"], **kwargs),
            copy_kwargs_=obj["copy_kwargs_"],
            reset_dct_=reset_dct_,
            reset_dct_copy_kwargs_=obj["reset_dct_copy_kwargs_"],
            frozen_keys_=obj["frozen_keys_"],
            readonly_=obj["readonly_"],
            nested_=obj["nested_"],
            convert_children_=obj["convert_children_"],
            as_attrs_=obj["as_attrs_"],
        )

    def load_update(
        self,
        path: tp.Optional[tp.PathLike] = None,
        clear: bool = False,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Load dumps from a file and update this instance in-place.

        !!! note
            Updates both the config properties and dictionary."""
        loaded = self.load(path=path, **kwargs)
        if clear:
            self.clear(force=True)
            self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        if nested is None:
            nested = self.nested_
        self.update(loaded, nested=nested, force=True)

    def prettify(
        self,
        with_params: bool = False,
        replace: tp.DictLike = None,
        path: str = None,
        htchar: str = "    ",
        lfchar: str = "\n",
        indent: int = 0,
    ) -> str:
        if with_params:
            dct = {
                **dict(self),
                **dict(
                    copy_kwargs_=self.copy_kwargs_,
                    reset_dct_=self.reset_dct_,
                    reset_dct_copy_kwargs_=self.reset_dct_copy_kwargs_,
                    frozen_keys_=self.frozen_keys_,
                    readonly_=self.readonly_,
                    nested_=self.nested_,
                    convert_children_=self.convert_children_,
                    as_attrs_=self.as_attrs_,
                ),
            }
        else:
            dct = dict(self)
        if all([isinstance(k, str) and k.isidentifier() for k in dct]):
            return prettify_inited(
                type(self),
                dct,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent,
            )
        return prettify_dict(self, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)

    def __eq__(self, other: tp.Any) -> bool:
        return checks.is_deep_equal(dict(self), dict(other))


class AtomicConfig(Config, atomic_dict):
    """Config that behaves like a single value when merging."""

    pass


class FrozenConfig(Config):
    """`Config` with `frozen_keys_` flag set to True."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if "frozen_keys_" in kwargs:
            del kwargs["frozen_keys_"]
        Config.__init__(
            self,
            *args,
            frozen_keys_=True,
            **kwargs,
        )


class ReadonlyConfig(Config):
    """`Config` with `readonly_` flag set to True."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if "readonly_" in kwargs:
            del kwargs["readonly_"]
        Config.__init__(
            self,
            *args,
            readonly_=True,
            **kwargs,
        )


class HybridConfig(Config):
    """`Config` with `copy_kwargs_` set to `copy_mode='hybrid'`."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        copy_kwargs_ = kwargs.pop("copy_kwargs_", dict())
        copy_kwargs_["copy_mode"] = "hybrid"
        Config.__init__(
            self,
            *args,
            copy_kwargs_=copy_kwargs_,
            **kwargs,
        )


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class Configured(Cacheable, Pickleable, Prettified):
    """Class with an initialization config.

    All subclasses of `Configured` are initialized using `Config`, which makes it easier to pickle.

    Settings are defined under `vectorbtpro._settings.configured`.

    !!! warning
        If any attribute has been overwritten that isn't listed in `Configured._writeable_attrs`,
        or if any `Configured.__init__` argument depends upon global defaults,
        their values won't be copied over. Make sure to pass them explicitly to
        make that the saved & loaded / copied instance is resilient to any changes in globals."""

    _settings_key: tp.ClassVar[tp.Optional[str]] = None
    """Key corresponding to this class in `vectorbtpro._settings`. 
    
    Multiple levels can be defined using the dot notation."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = None
    """Set of expected keys."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = None
    """Set of writeable attributes that will be saved/copied along with the config."""

    def __init__(self, **config) -> None:
        from vectorbtpro._settings import settings

        configured_cfg = settings["configured"]

        check_expected_keys_ = config.get("check_expected_keys_", None)
        if self._expected_keys is None:
            check_expected_keys_ = False
        if check_expected_keys_ is None:
            check_expected_keys_ = configured_cfg["check_expected_keys_"]
        if check_expected_keys_:
            if isinstance(check_expected_keys_, bool):
                check_expected_keys_ = "raise"
            keys_diff = list(set(config.keys()).difference(self._expected_keys))
            if len(keys_diff) > 0:
                checks.assert_in(check_expected_keys_, ("warn", "raise"))
                if check_expected_keys_ == "warn":
                    warnings.warn(f"{type(self).__name__} doesn't expect arguments {keys_diff}", stacklevel=2)
                else:
                    raise ValueError(f"{type(self).__name__} doesn't expect arguments {keys_diff}")

        self._config = Config(**merge_dicts(configured_cfg["config"], config))

        Cacheable.__init__(self)

    @property
    def config(self) -> Config:
        """Initialization config."""
        return self._config

    @class_or_instancemethod
    def get_writeable_attrs(cls_or_self) -> tp.Optional[tp.Set[str]]:
        """Get set of attributes that are writeable by this class or by any of its base classes."""
        if isinstance(cls_or_self, type):
            cls = cls_or_self
        else:
            cls = type(cls_or_self)
        writeable_attrs = set()
        for cls in inspect.getmro(cls):
            if issubclass(cls, Configured) and cls._writeable_attrs is not None:
                writeable_attrs |= cls._writeable_attrs
        return writeable_attrs

    def replace(
        self: ConfiguredT,
        copy_mode_: tp.Optional[str] = None,
        nested_: tp.Optional[bool] = None,
        cls_: tp.Optional[type] = None,
        copy_writeable_attrs_: tp.Optional[bool] = None,
        **new_config,
    ) -> ConfiguredT:
        """Create a new instance by copying and (optionally) changing the config.

        !!! warning
            This operation won't return a copy of the instance but a new instance
            initialized with the same config and writeable attributes (or their copy, depending on `copy_mode`)."""
        if cls_ is None:
            cls_ = type(self)
        if copy_writeable_attrs_ is None:
            copy_writeable_attrs_ = cls_ is type(self)
        new_config = self.config.merge_with(new_config, copy_mode=copy_mode_, nested=nested_)
        new_instance = cls_(**new_config)
        if copy_writeable_attrs_:
            for attr in self.get_writeable_attrs():
                attr_obj = getattr(self, attr)
                if isinstance(attr_obj, Config):
                    attr_obj = attr_obj.copy(copy_mode=copy_mode_, nested=nested_)
                else:
                    if copy_mode_ is not None:
                        if copy_mode_ == "hybrid":
                            attr_obj = copy(attr_obj)
                        elif copy_mode_ == "deep":
                            attr_obj = deepcopy(attr_obj)
                setattr(new_instance, attr, attr_obj)
        return new_instance

    def copy(
        self: ConfiguredT,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        cls: tp.Optional[type] = None,
    ) -> ConfiguredT:
        """Create a new instance by copying the config.

        See `Configured.replace`."""
        return self.replace(copy_mode_=copy_mode, nested_=nested, cls_=cls)

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        config_dumps = self.config.dumps(**kwargs)
        attr_dct = PickleableDict({attr: getattr(self, attr) for attr in self.get_writeable_attrs()})
        attr_dct_dumps = attr_dct.dumps(**kwargs)
        return pickle.dumps((config_dumps, attr_dct_dumps), **kwargs)

    @classmethod
    def loads(cls: tp.Type[ConfiguredT], dumps: bytes, **kwargs) -> ConfiguredT:
        """Unpickle from bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import

        warn_cannot_import("dill")
        try:
            import dill as pickle
        except ImportError:
            import pickle

        config_dumps, attr_dct_dumps = pickle.loads(dumps, **kwargs)
        config = Config.loads(config_dumps, **kwargs)
        attr_dct = PickleableDict.loads(attr_dct_dumps, **kwargs)
        new_instance = cls(**config)
        for attr, obj in attr_dct.items():
            setattr(new_instance, attr, obj)
        return new_instance

    def __eq__(self, other: tp.Any) -> bool:
        """Objects are equal if their configs and writeable attributes are equal."""
        if type(self) != type(other):
            return False
        if self.get_writeable_attrs() != other.get_writeable_attrs():
            return False
        for attr in self.get_writeable_attrs():
            if not checks.is_deep_equal(getattr(self, attr), getattr(other, attr)):
                return False
        return self.config == other.config

    def update_config(self, *args, **kwargs) -> None:
        """Force-update the config."""
        self.config.update(*args, **kwargs, force=True)

    @classmethod
    def get_settings(cls) -> dict:
        """Get class-related settings from `vectorbtpro._settings`."""
        from vectorbtpro._settings import settings

        final_cls_cfg = None
        for c in cls.__mro__[::-1]:
            if hasattr(c, "_settings_key"):
                c_settings_key = getattr(c, "_settings_key")
                if c_settings_key is not None:
                    c_cfg = get_dict_item(settings, c_settings_key)
                    final_cls_cfg = merge_dicts(final_cls_cfg, c_cfg)
        return final_cls_cfg

    @classmethod
    def set_settings(cls, **kwargs) -> None:
        """Set class-related settings in `vectorbtpro._settings`."""
        from vectorbtpro._settings import settings

        if cls._settings_key is None:
            raise ValueError(f"No settings associated with the class {cls.__name__}")
        cls_cfg = get_dict_item(settings, cls._settings_key)
        for k, v in kwargs.items():
            if k not in cls_cfg:
                raise KeyError(f"Invalid key '{k}'")
            if isinstance(cls_cfg[k], dict) and isinstance(v, dict):
                cls_cfg[k] = merge_dicts(cls_cfg[k], v)
            else:
                cls_cfg[k] = v

    @classmethod
    def reset_settings(cls) -> None:
        """Reset class-related settings in `vectorbtpro._settings`."""
        from vectorbtpro._settings import settings

        if cls._settings_key is None:
            raise ValueError(f"No settings associated with the class {cls.__name__}")
        cls_cfg = get_dict_item(settings, cls._settings_key)
        cls_cfg.reset(force=True)

    def prettify(self, **kwargs) -> str:
        return "%s(%s)" % (
            type(self).__name__,
            self.config.prettify(**kwargs)[len(type(self.config).__name__) + 1 : -1],
        )
