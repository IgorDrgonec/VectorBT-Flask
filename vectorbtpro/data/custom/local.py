# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `LocalData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "LocalData",
]

__pdoc__ = {}


class LocalData(CustomData):
    """Data class for fetching local data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.local")
