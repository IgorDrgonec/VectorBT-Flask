# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for pickling."""

import humanize
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes."""

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import
        warn_cannot_import('dill')
        try:
            import dill as pickle
        except ImportError:
            import pickle

        return pickle.dumps(self, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableT], dumps: bytes, **kwargs) -> PickleableT:
        """Unpickle from bytes."""
        from vectorbtpro.utils.opt_packages import warn_cannot_import
        warn_cannot_import('dill')
        try:
            import dill as pickle
        except ImportError:
            import pickle

        return pickle.loads(dumps, **kwargs)

    def save(self, path: tp.PathLike, mkdir_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """Save dumps to a file.

        If `path` has no suffix, adds the suffix '.pickle'."""
        path = Path(path)
        if mkdir_kwargs is None:
            mkdir_kwargs = {}
        if path.suffix == '':
            path = path.with_suffix('.pickle')
        check_mkdir(path.parent, **mkdir_kwargs)
        dumps = self.dumps(**kwargs)
        with open(path, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls: tp.Type[PickleableT], path: tp.PathLike, **kwargs) -> PickleableT:
        """Load dumps from a file and create new instance.

        If `path` has no suffix and doesn't exist, checks whether there is the same path
        but with the suffix '.pickle' or '.pkl'."""
        path = Path(path)
        if path.suffix == '' and not path.exists():
            if path.with_suffix('.pickle').exists():
                path = path.with_suffix('.pickle')
            elif path.with_suffix('.pkl').exists():
                path = path.with_suffix('.pkl')
        with open(path, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)

    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get size of this object."""
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()
