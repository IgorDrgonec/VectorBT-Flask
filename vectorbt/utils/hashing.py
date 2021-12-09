# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for hashing."""

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cachedproperty


class Hashable:
    """Hashable class."""

    @staticmethod
    def get_hash(*args, **kwargs) -> int:
        """Static method to get the hash of the instance based on its arguments."""
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        """Key that can be used for hashing the instance."""
        raise NotImplementedError

    @cachedproperty
    def hash(self) -> int:
        """Hash of the instance."""
        return hash(self.hash_key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, type(self)):
            return self.hash_key == other.hash_key
        raise NotImplementedError
