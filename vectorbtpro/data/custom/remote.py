# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `RemoteData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "RemoteData",
]

__pdoc__ = {}


class RemoteData(CustomData):
    """Data class for fetching remote data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.remote")
