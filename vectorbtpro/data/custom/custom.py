# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `CustomData`."""

import re
import fnmatch

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data

__all__ = [
    "CustomData",
]

__pdoc__ = {}


class CustomData(Data):
    """Data class for fetching custom data."""

    _setting_keys: tp.SettingsKeys = dict(custom=None)

    @classmethod
    def get_custom_settings(cls) -> dict:
        """`CustomData.get_settings` with `key_id="custom"`."""
        return cls.get_settings(key_id="custom")

    @classmethod
    def set_custom_settings(cls, **kwargs) -> None:
        """`CustomData.set_settings` with `key_id="custom"`."""
        cls.set_settings(key_id="custom", **kwargs)

    @classmethod
    def reset_custom_settings(cls) -> None:
        """`CustomData.reset_settings` with `key_id="custom"`."""
        cls.reset_settings(key_id="custom")

    @staticmethod
    def key_match(key: str, pattern: str, use_regex: bool = False):
        """Return whether key matches pattern.

        If `use_regex` is True, checks against a regular expression.
        Otherwise, checks against a glob-style pattern."""
        if use_regex:
            return re.match(pattern, key)
        return re.match(fnmatch.translate(pattern), key)
