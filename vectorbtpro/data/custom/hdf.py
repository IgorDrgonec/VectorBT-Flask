# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `HDFData`."""

from pathlib import Path, PurePath
from glob import glob
import re

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    to_tzaware_timestamp,
    to_naive_timestamp,
)
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.data.custom.file import FileData

__all__ = [
    "HDFData",
]

__pdoc__ = {}


class HDFPathNotFoundError(Exception):
    """Gets raised if the path to an HDF file could not be found."""

    pass


class HDFKeyNotFoundError(Exception):
    """Gets raised if the key to an HDF object could not be found."""

    pass


HDFDataT = tp.TypeVar("HDFDataT", bound="HDFData")


class HDFData(FileData):
    """Data class for fetching HDF data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.hdf")

    @classmethod
    def list_keys(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[tp.Key]:
        """List all features or symbols under a path."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "**" / "*.h5"
        return list(map(str, cls.match_path(path, **match_path_kwargs)))

    @classmethod
    def list_features(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[tp.Feature]:
        """List all features under a path."""
        return cls.list_keys(path=path, **match_path_kwargs)

    @classmethod
    def list_symbols(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[tp.Symbol]:
        """List all symbols under a path."""
        return cls.list_keys(path=path, **match_path_kwargs)

    @classmethod
    def split_hdf_path(
        cls,
        path: tp.PathLike,
        key: tp.Optional[str] = None,
        _full_path: tp.Optional[Path] = None,
    ) -> tp.Tuple[Path, tp.Optional[str]]:
        """Split the path to an HDF object into the path to the file and the key."""
        path = Path(path)
        if _full_path is None:
            _full_path = path
        if path.exists():
            if path.is_dir():
                raise HDFPathNotFoundError(f"No HDF files could be matched with {_full_path}")
            return path, key
        new_path = path.parent
        if key is None:
            new_key = path.name
        else:
            new_key = str(Path(path.name) / key)
        return cls.split_hdf_path(new_path, new_key, _full_path=_full_path)

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        recursive: bool = True,
        **kwargs,
    ) -> tp.List[Path]:
        """Override `FileData.match_path` to return a list of HDF paths
        (path to file + key) matching a path."""
        path = Path(path)
        if path.exists():
            if path.is_dir():
                sub_paths = [p for p in path.iterdir() if p.is_file()]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
            else:
                with pd.HDFStore(str(path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                key_paths = [path / k for k in keys]
        else:
            try:
                file_path, key = cls.split_hdf_path(path)
                with pd.HDFStore(str(file_path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                if key is None:
                    key_paths = [file_path / k for k in keys]
                elif key in keys:
                    key_paths = [file_path / key]
                else:
                    matching_keys = []
                    for k in keys:
                        if k.startswith(key) or PurePath("/" + str(k)).match("/" + str(key)):
                            matching_keys.append(k)
                    if len(matching_keys) == 0:
                        raise HDFKeyNotFoundError(f"No HDF keys could be matched with {key}")
                    key_paths = [file_path / k for k in matching_keys]
            except HDFPathNotFoundError:
                sub_paths = list([Path(p) for p in glob(str(path), recursive=recursive)])
                if len(sub_paths) == 0 and re.match(r".+\..+", str(path)):
                    base_path = None
                    base_ended = False
                    key_path = None
                    for part in path.parts:
                        part = Path(part)
                        if base_ended:
                            if key_path is None:
                                key_path = part
                            else:
                                key_path /= part
                        else:
                            if re.match(r".+\..+", str(part)):
                                base_ended = True
                            if base_path is None:
                                base_path = part
                            else:
                                base_path /= part
                    sub_paths = list([Path(p) for p in glob(str(base_path), recursive=recursive)])
                    if key_path is not None:
                        sub_paths = [p / key_path for p in sub_paths]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
        if match_regex is not None:
            key_paths = [p for p in key_paths if re.match(match_regex, str(p))]
        if sort_paths:
            key_paths = sorted(key_paths)
        return key_paths

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_hdf_kwargs,
    ) -> tp.KeyData:
        """Load the HDF object for a feature or symbol.

        Args:
            key (hashable): Feature or symbol.
            path (str): Path.

                Will be resolved with `HDFData.split_hdf_path`.

                If `path` is None, uses `key` as the path to the HDF file.
            start (any): Start datetime.

                Will extract the object's index and compare the index to the date.
                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Can only be used if the object was saved in the table format!
            end (any): End datetime.

                Will extract the object's index and compare the index to the date.
                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Can only be used if the object was saved in the table format!
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (int): Start row (inclusive).

                Will use it when querying index as well.
            end_row (int): End row (exclusive).

                Will use it when querying index as well.
            chunk_func (callable): Function to select and concatenate chunks from `TableIterator`.

                Gets called only if `iterator` or `chunksize` are set.
            **read_hdf_kwargs: Keyword arguments passed to `pd.read_hdf`.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments.

        For defaults, see `custom.hdf` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        from pandas.io.pytables import TableIterator

        hdf_cfg = cls.get_settings(key_id="custom")

        if start is None:
            start = hdf_cfg["start"]
        if end is None:
            end = hdf_cfg["end"]
        if tz is None:
            tz = hdf_cfg["tz"]
        if start_row is None:
            start_row = hdf_cfg["start_row"]
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = hdf_cfg["end_row"]
        read_hdf_kwargs = merge_dicts(hdf_cfg["read_hdf_kwargs"], read_hdf_kwargs)

        if path is None:
            path = key
        path = Path(path)
        file_path, key = cls.split_hdf_path(path)

        if start is not None or end is not None:
            hdf_store_arg_names = get_func_arg_names(pd.HDFStore.__init__)
            hdf_store_kwargs = dict()
            for k, v in read_hdf_kwargs.items():
                if k in hdf_store_arg_names:
                    hdf_store_kwargs[k] = v
            with pd.HDFStore(str(file_path), mode="r", **hdf_store_kwargs) as store:
                index = store.select_column(key, "index", start=start_row, stop=end_row)
            if not isinstance(index, pd.Index):
                index = pd.Index(index)
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if tz is None:
                tz = index.tzinfo
            if index.tzinfo is not None:
                if start is not None:
                    start = to_tzaware_timestamp(start, naive_tz=tz, tz=index.tzinfo)
                if end is not None:
                    end = to_tzaware_timestamp(end, naive_tz=tz, tz=index.tzinfo)
            else:
                if start is not None:
                    start = to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= index >= start
            if end is not None:
                mask &= index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            start_row += mask_indices[0]
            end_row = start_row + mask_indices[-1] - mask_indices[0] + 1

        obj = pd.read_hdf(file_path, key=key, start=start_row, stop=end_row, **read_hdf_kwargs)
        if isinstance(obj, TableIterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tzinfo
        return obj, dict(last_row=start_row + len(obj.index) - 1, tz_convert=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Load the HDF object for a feature.

        Calls `HDFData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Load the HDF object for a symbol.

        Calls `HDFData.fetch_key`."""
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update data for a feature or symbol."""
        if key_is_feature:
            fetch_kwargs = self.select_feature_kwargs(key, self.fetch_kwargs)
        else:
            fetch_kwargs = self.select_symbol_kwargs(key, self.fetch_kwargs)
        fetch_kwargs["start_row"] = self.returned_kwargs[key]["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Update data for a feature.

        Calls `HDFData.update_key` with `key_is_feature=True`."""
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Calls `HDFData.update_key` with `key_is_feature=False`."""
        return self.update_key(symbol, key_is_feature=False, **kwargs)
