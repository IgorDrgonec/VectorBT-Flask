# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `CSVData`."""

from pathlib import Path

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    to_tzaware_timestamp,
    to_naive_timestamp,
)
from vectorbtpro.data.custom.file import FileData

__all__ = [
    "CSVData",
]

__pdoc__ = {}

CSVDataT = tp.TypeVar("CSVDataT", bound="CSVData")


class CSVData(FileData):
    """Data class for fetching CSV data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.csv")

    @classmethod
    def list_keys(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[tp.Key]:
        """List all features or symbols under a path."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "**" / "*.csv"
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
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        header: tp.Optional[tp.MaybeSequence[int]] = None,
        index_col: tp.Optional[int] = None,
        parse_dates: tp.Optional[bool] = None,
        squeeze: tp.Optional[bool] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_csv_kwargs,
    ) -> tp.KeyData:
        """Load a CSV file for a feature or symbol.

        Args:
            key (hashable): Feature or symbol.
            path (str): Path.

                If `path` is None, uses `key` as the path to the CSV file.
            start (any): Start datetime.

                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (any): End datetime.

                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (int): Start row (inclusive).

                Must exclude header rows.
            end_row (int): End row (exclusive).

                Must exclude header rows.
            header (int or sequence of int): See `pd.read_csv`.
            index_col (int): See `pd.read_csv`.
            parse_dates (bool): See `pd.read_csv`.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            chunk_func (callable): Function to select and concatenate chunks from `TextFileReader`.

                Gets called only if `iterator` or `chunksize` are set.
            **read_csv_kwargs: Keyword arguments passed to `pd.read_csv`.

        `skiprows` and `nrows` will be automatically calculated based on `start_row` and `end_row`.

        When either `start` or `end` is provided, will fetch the entire data first and filter it thereafter.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for other arguments.

        For defaults, see `custom.csv` in `vectorbtpro._settings.data`."""
        from pandas.io.parsers import TextFileReader
        from pandas.api.types import is_object_dtype

        csv_cfg = cls.get_settings(key_id="custom")

        if start is None:
            start = csv_cfg["start"]
        if end is None:
            end = csv_cfg["end"]
        if tz is None:
            tz = csv_cfg["tz"]
        if start_row is None:
            start_row = csv_cfg["start_row"]
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = csv_cfg["end_row"]
        if header is None:
            header = csv_cfg["header"]
        if index_col is None:
            index_col = csv_cfg["index_col"]
        if index_col is False:
            index_col = None
        if parse_dates is None:
            parse_dates = csv_cfg["parse_dates"]
        if squeeze is None:
            squeeze = csv_cfg["squeeze"]
        read_csv_kwargs = merge_dicts(csv_cfg["read_csv_kwargs"], read_csv_kwargs)

        if path is None:
            path = key
        if isinstance(header, int):
            header = [header]
        header_rows = header[-1] + 1
        start_row += header_rows
        if end_row is not None:
            end_row += header_rows
        skiprows = range(header_rows, start_row)
        if end_row is not None:
            nrows = end_row - start_row
        else:
            nrows = None

        sep = read_csv_kwargs.pop("sep", None)
        if isinstance(path, (str, Path)):
            try:
                _path = Path(path)
                if _path.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
            except Exception as e:
                pass
        if sep is None:
            sep = ","

        obj = pd.read_csv(
            path,
            sep=sep,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            skiprows=skiprows,
            nrows=nrows,
            **read_csv_kwargs,
        )

        if isinstance(obj, TextFileReader):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        if index_col is not None and parse_dates and is_object_dtype(obj.index.dtype):
            obj.index = pd.to_datetime(obj.index, utc=True)
            if tz is not None:
                obj.index = obj.index.tz_convert(tz)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tzinfo
        if start is not None or end is not None:
            if not isinstance(obj.index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if obj.index.tzinfo is not None:
                if start is not None:
                    start = to_tzaware_timestamp(start, naive_tz=tz, tz=obj.index.tzinfo)
                if end is not None:
                    end = to_tzaware_timestamp(end, naive_tz=tz, tz=obj.index.tzinfo)
            else:
                if start is not None:
                    start = to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= obj.index >= start
            if end is not None:
                mask &= obj.index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            obj = obj.iloc[mask_indices[0] : mask_indices[-1] + 1]
            start_row += mask_indices[0]
        return obj, dict(last_row=start_row - header_rows + len(obj.index) - 1, tz_convert=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Load the CSV file for a feature.

        Calls `CSVData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Load the CSV file for a symbol.

        Calls `CSVData.fetch_key`."""
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

        Calls `CSVData.update_key` with `key_is_feature=True`."""
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Calls `CSVData.update_key` with `key_is_feature=False`."""
        return self.update_key(symbol, key_is_feature=False, **kwargs)
