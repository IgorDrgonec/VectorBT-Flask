# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `RemoteData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "RemoteData",
]

__pdoc__ = {}

RemoteDataT = tp.TypeVar("RemoteDataT", bound="RemoteData")


class RemoteData(CustomData):
    """Data class for fetching remote data.

    Remote data usually has arguments such as `start`, `end`, and `timeframe`.

    Overrides `vectorbtpro.data.base.Data.update_symbol` to update data based on the `start` argument."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.remote")
