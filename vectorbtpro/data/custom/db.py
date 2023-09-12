# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `DBData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.local import LocalData

__all__ = [
    "DBData",
]

__pdoc__ = {}


class DBData(LocalData):
    """Data class for fetching database data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.db")
