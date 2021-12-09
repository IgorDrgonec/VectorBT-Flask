# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for pickling."""

import humanize

from vectorbt import _typing as tp

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes."""

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        from vectorbt.opt_packages import warn_cannot_import
        warn_cannot_import('dill')
        try:
            import dill as pickle
        except ImportError:
            import pickle

        return pickle.dumps(self, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableT], dumps: bytes, **kwargs) -> PickleableT:
        """Unpickle from bytes."""
        from vectorbt.opt_packages import warn_cannot_import
        warn_cannot_import('dill')
        try:
            import dill as pickle
        except ImportError:
            import pickle

        return pickle.loads(dumps, **kwargs)

    def save(self, fname: tp.PathLike, **kwargs) -> None:
        """Save dumps to a file."""
        dumps = self.dumps(**kwargs)
        with open(fname, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls: tp.Type[PickleableT], fname: tp.PathLike, **kwargs) -> PickleableT:
        """Load dumps from a file and create new instance."""
        with open(fname, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)

    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get size of this object."""
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()
