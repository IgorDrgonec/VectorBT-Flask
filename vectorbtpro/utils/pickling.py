# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for pickling."""

import attr
import humanize
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.module_ import find_class

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


def dumps(obj: tp.Any, **kwargs) -> bytes:
    """Pickle an object."""
    from vectorbtpro.utils.opt_packages import warn_cannot_import

    warn_cannot_import("dill")
    try:
        import dill as pickle
    except ImportError:
        import pickle

    return pickle.dumps(obj, **kwargs)


def loads(bytes_: bytes, **kwargs) -> tp.Any:
    """Unpickle an object."""
    from vectorbtpro.utils.opt_packages import warn_cannot_import

    warn_cannot_import("dill")
    try:
        import dill as pickle
    except ImportError:
        import pickle

    return pickle.loads(bytes_, **kwargs)


def save_bytes(bytes_: bytes, path: tp.PathLike, mkdir_kwargs: tp.KwargsLike = None) -> None:
    """Save bytes to a file."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".pickle")
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    with open(path, "wb") as f:
        f.write(bytes_)


def load_bytes(path: tp.PathLike) -> bytes:
    """Load bytes from a file.

    If `path` has no suffix and doesn't exist, checks whether there is the same path
    but with the suffix '.pickle' or '.pkl'."""
    path = Path(path)
    if path.suffix == "" and not path.exists():
        if path.with_suffix(".pickle").exists():
            path = path.with_suffix(".pickle")
        elif path.with_suffix(".pkl").exists():
            path = path.with_suffix(".pkl")
    with open(path, "rb") as f:
        bytes_ = f.read()
    return bytes_


def save(obj: object, path: tp.Optional[tp.PathLike] = None, mkdir_kwargs: tp.KwargsLike = None, **kwargs) -> None:
    """Pickle to a file."""
    if path is None:
        path = type(obj).__name__
    bytes_ = dumps(obj, **kwargs)
    save_bytes(bytes_, path, mkdir_kwargs=mkdir_kwargs)


def load(path: tp.PathLike, **kwargs) -> object:
    """Unpickle from a file."""
    bytes_ = load_bytes(path)
    return loads(bytes_, **kwargs)


@attr.s(frozen=True)
class PRecState:
    """Class that represents a state used to reconstruct an instance."""

    init_args: tp.Args = attr.ib(default=attr.Factory(tuple))
    """Positional arguments used in initialization."""

    init_kwargs: tp.Kwargs = attr.ib(default=attr.Factory(dict))
    """Keyword arguments used in initialization."""

    attr_dct: tp.Kwargs = attr.ib(default=attr.Factory(dict))
    """Dictionary with names and values of writeable attributes."""


@attr.s(frozen=True)
class PRecInfo:
    """Class that represents information needed to reconstruct an instance."""

    id_: tp.Hashable = attr.ib()
    """Identifier."""

    cls: tp.Type = attr.ib()
    """Class."""

    modify_state: tp.Optional[tp.Callable[[PRecState], PRecState]] = attr.ib(default=None)
    """Callback to modify the reconstruction state."""

    def register(self) -> None:
        """Register self in `prec_info_registry`."""
        prec_info_registry[self.id_] = self


prec_info_registry = {}
"""Registry with instances of `PRecInfo` keyed by `PRecInfo.id_`.

Populate with the required information if any instance cannot be unpickled."""


def reconstruct(cls: tp.Union[tp.Hashable, tp.Type], prec_state: PRecState) -> object:
    """Reconstruct an instance using a class and a reconstruction state."""
    found_prec = False
    if not isinstance(cls, type):
        cls_id = cls
        if cls_id in prec_info_registry:
            found_prec = True
            cls = prec_info_registry[cls_id].cls
            modify_state = prec_info_registry[cls_id].modify_state
            if modify_state is not None:
                prec_state = modify_state(prec_state)
    if not isinstance(cls, type):
        if isinstance(cls, str):
            cls_name = cls
            cls = find_class(cls_name)
            if cls is None:
                cls = cls_name
    if not isinstance(cls, type):
        raise ValueError(f"Please register an instance of PRecInfo for '{cls}'")
    if not found_prec:
        class_path = type(cls).__module__ + "." + type(cls).__name__
        if class_path in prec_info_registry:
            cls = prec_info_registry[class_path].cls
            modify_state = prec_info_registry[class_path].modify_state
            if modify_state is not None:
                prec_state = modify_state(prec_state)
    obj = cls(*prec_state.init_args, **prec_state.init_kwargs)
    for k, v in prec_state.attr_dct.items():
        setattr(obj, k, v)
    return obj


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes.

    If any subclass cannot be pickled, override the `Pickleable.prec_state` property
    to return an instance of `PRecState` to be used in reconstruction. If the class definition cannot
    be pickled (e.g., created dynamically), override its `_rec_id` with an arbitrary id (such as `123456789`),
    dump/save the class, and before loading, map this id to the class in `rec_id_map`. This will use the
    mapped class to construct a new instance."""

    _prec_id: tp.ClassVar[tp.Optional[tp.Hashable]] = None
    """Reconstruction id."""

    @classmethod
    def file_exists(cls, path: tp.Optional[tp.PathLike] = None) -> bool:
        """Return whether a file already exists."""
        if path is None:
            path = cls.__name__
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".pickle")
        return path.exists()

    def dumps(self, prec_state_only: bool = False, **kwargs) -> bytes:
        """Pickle the instance.

        Optionally, you can set `prec_state_only` to True if the instance will be later unpickled
        directly by the class."""
        if prec_state_only:
            prec_state = self.prec_state
            if prec_state is None:
                raise ValueError("Reconstruction state is None")
            return dumps(prec_state, **kwargs)
        return dumps(self, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableT], bytes_: bytes, **kwargs) -> PickleableT:
        """Unpickle an instance."""
        obj = loads(bytes_, **kwargs)
        if isinstance(obj, PRecState):
            obj = reconstruct(cls, obj)
        if not isinstance(obj, cls):
            raise TypeError(f"Unpickled object must be an instance of {cls}")
        return obj

    def save(self, path: tp.Optional[tp.PathLike] = None, mkdir_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """Pickle the instance to a file.

        If `path` has no suffix, adds the suffix '.pickle'."""
        if path is None:
            path = type(self).__name__
        bytes_ = self.dumps(**kwargs)
        save_bytes(bytes_, path, mkdir_kwargs=mkdir_kwargs)

    @classmethod
    def load(cls: tp.Type[PickleableT], path: tp.Optional[tp.PathLike] = None, **kwargs) -> PickleableT:
        """Unpickle from a file and create a new instance.

        If `path` has no suffix and doesn't exist, checks whether there is the same path
        but with the suffix '.pickle' or '.pkl'."""
        if path is None:
            path = cls.__name__
        bytes_ = load_bytes(path)
        return cls.loads(bytes_, **kwargs)

    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get size of this object."""
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()

    @property
    def prec_state(self) -> tp.Optional[PRecState]:
        """Reconstruction state of the type `PRecState`."""
        return None

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        prec_state = self.prec_state
        if prec_state is None:
            return object.__reduce__(self)
        if self._prec_id is None:
            class_path = type(self).__module__ + "." + type(self).__name__
            if find_class(class_path) is not None:
                cls = class_path
            else:
                cls = type(self)
        else:
            cls = self._prec_id
        return reconstruct, (cls, prec_state)
