# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Custom data source classes.

!!! note
    Use absolute start and end dates instead of relative ones when fetching multiple
    symbols of data: some symbols may take a considerable amount of time to fetch
    such that they may shift the time period for the symbols coming next.

    This happens when relative dates are parsed in `vectorbtpro.data.base.Data.fetch_symbol`
    instead of parsing them once and for all symbols in `vectorbtpro.data.base.Data.fetch`."""

import time
import traceback
import warnings
from functools import wraps
from pathlib import Path, PurePath
from glob import glob
import re

import pandas as pd
from pandas.io.parsers import TextFileReader
from pandas.io.pytables import TableIterator

from vectorbtpro import _typing as tp
from vectorbtpro.data import nb
from vectorbtpro.data.base import Data, symbol_dict
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    get_utc_tz,
    get_local_tz,
    to_tzaware_datetime,
    datetime_to_ms,
)
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.random_ import set_seed

try:
    from binance.client import Client as BinanceClientT
except ImportError:
    BinanceClientT = tp.Any
try:
    from ccxt.base.exchange import Exchange as ExchangeT
except ImportError:
    ExchangeT = tp.Any
try:
    from alpaca_trade_api.rest import REST as AlpacaClientT
except ImportError:
    AlpacaClientT = tp.Any

__all__ = [
    "SyntheticData",
    "RandomData",
    "GBMData",
    "LocalData",
    "CSVData",
    "HDFData",
    "RemoteData",
    "YFData",
    "BinanceData",
    "CCXTData",
    "AlpacaData",
]

# ############# Synthetic ############# #


class SyntheticData(Data):
    """Subclass of `Data` for synthetically generated data.

    Exposes an abstract class method `SyntheticData.generate_symbol`.
    Everything else is taken care of."""

    @classmethod
    def generate_symbol(cls, symbol: tp.Symbol, index: tp.Index, **kwargs) -> tp.SeriesFrame:
        """Abstract method to generate data of a symbol."""
        raise NotImplementedError

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        freq: tp.Union[None, str, pd.DateOffset] = None,
        date_range_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to generate a symbol.

        Generates datetime index and passes it to `SyntheticData.generate_symbol` to fill
        the Series/DataFrame with generated data.

        For defaults, see `custom.synthetic` in `vectorbtpro._settings.data`."""
        from vectorbtpro._settings import settings

        synthetic_cfg = settings["data"]["custom"]["synthetic"]

        if start is None:
            start = synthetic_cfg["start"]
        if end is None:
            end = synthetic_cfg["end"]
        if freq is None:
            freq = synthetic_cfg["freq"]
        date_range_kwargs = merge_dicts(synthetic_cfg["date_range_kwargs"], date_range_kwargs)

        index = pd.date_range(
            start=to_tzaware_datetime(start, tz=get_utc_tz()),
            end=to_tzaware_datetime(end, tz=get_utc_tz()),
            freq=freq,
            **date_range_kwargs,
        )
        if len(index) == 0:
            raise ValueError("Date range is empty")
        return cls.generate_symbol(symbol, index, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class RandomData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_random_data_nb`."""

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        num_paths: tp.Optional[int] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            num_paths (int): Number of generated paths (columns in our case).
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        For defaults, see `custom.random` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        from vectorbtpro._settings import settings

        random_cfg = settings["data"]["custom"]["random"]

        if num_paths is None:
            num_paths = random_cfg["num_paths"]
        if start_value is None:
            start_value = random_cfg["start_value"]
        if mean is None:
            mean = random_cfg["mean"]
        if std is None:
            std = random_cfg["std"]
        if seed is None:
            seed = random_cfg["seed"]
        if seed is not None:
            set_seed(seed)
        if jitted is None:
            jitted = random_cfg["jitted"]

        func = jit_reg.resolve_option(nb.generate_random_data_nb, jitted)
        out = func((len(index), num_paths), start_value, mean, std)

        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name="path")
        return pd.DataFrame(out, index=index, columns=columns)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol].iloc[-2]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)


class GBMData(RandomData):
    """`RandomData` for data generated using `vectorbtpro.data.nb.generate_gbm_data_nb`."""

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        num_paths: tp.Optional[int] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            num_paths (int): Number of generated paths (columns in our case).
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            dt (float): Time change (one period of time).
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        For defaults, see `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        from vectorbtpro._settings import settings

        gbm_cfg = settings["data"]["custom"]["gbm"]

        if num_paths is None:
            num_paths = gbm_cfg["num_paths"]
        if start_value is None:
            start_value = gbm_cfg["start_value"]
        if mean is None:
            mean = gbm_cfg["mean"]
        if std is None:
            std = gbm_cfg["std"]
        if dt is None:
            dt = gbm_cfg["dt"]
        if seed is None:
            seed = gbm_cfg["seed"]
        if seed is not None:
            set_seed(seed)
        if jitted is None:
            jitted = gbm_cfg["jitted"]

        func = jit_reg.resolve_option(nb.generate_gbm_data_nb, jitted)
        out = func((len(index), num_paths), start_value, mean, std, dt)

        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name="path")
        return pd.DataFrame(out, index=index, columns=columns)


# ############# Local ############# #

LocalDataT = tp.TypeVar("LocalDataT", bound="LocalData")


class LocalData(Data):
    """Subclass of `Data` for local data."""

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        **kwargs,
    ) -> tp.List[Path]:
        """Get the list of all paths matching a path."""
        path = Path(path)
        if path.exists():
            if path.is_dir():
                sub_paths = [p for p in path.iterdir() if p.is_file()]
            else:
                sub_paths = [path]
        else:
            sub_paths = list([Path(p) for p in glob(str(path), recursive=True)])
        if match_regex is not None:
            sub_paths = [p for p in sub_paths if re.match(match_regex, str(p))]
        if sort_paths:
            sub_paths = sorted(sub_paths)
        return sub_paths

    @classmethod
    def path_to_symbol(cls, path: tp.PathLike, **kwargs) -> str:
        """Convert a path into a symbol."""
        return Path(path).stem

    @classmethod
    def fetch(
        cls: tp.Type[LocalDataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        paths: tp.Any = None,
        match_paths: tp.Optional[bool] = None,
        match_regex: tp.Optional[str] = None,
        sort_paths: tp.Optional[bool] = None,
        match_path_kwargs: tp.KwargsLike = None,
        path_to_symbol_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> LocalDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to take care of paths.

        Use either `symbols` or `path` to specify the path to one or multiple files.
        Allowed are paths in string or `pathlib.Path` format, or string expressions accepted by `glob.glob`.

        Set `match_paths` to False to not parse paths and behave like a regular
        `vectorbtpro.data.base.Data` instance.

        For defaults, see `custom.local` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro._settings import settings

        local_cfg = settings["data"]["custom"]["local"]

        if match_paths is None:
            match_paths = local_cfg["match_paths"]
        if match_regex is None:
            match_regex = local_cfg["match_regex"]
        if sort_paths is None:
            sort_paths = local_cfg["sort_paths"]

        if match_paths:
            sync = False
            if paths is None:
                paths = symbols
                sync = True
            elif symbols is None:
                sync = True
            if paths is None:
                raise ValueError("At least symbols or paths must be set")
            if match_path_kwargs is None:
                match_path_kwargs = {}
            if path_to_symbol_kwargs is None:
                path_to_symbol_kwargs = {}

            single_symbol = False
            if isinstance(symbols, (str, Path)):
                # Single symbol
                symbols = [symbols]
                single_symbol = True

            single_path = False
            if isinstance(paths, (str, Path)):
                # Single path
                paths = [paths]
                single_path = True
                if sync:
                    single_symbol = True

            if isinstance(paths, symbol_dict):
                # Dict of path per symbol
                if sync:
                    symbols = list(paths.keys())
                elif len(symbols) != len(paths):
                    raise ValueError("The number of symbols must be equal to the number of matched paths")
            elif checks.is_iterable(paths) or checks.is_sequence(paths):
                # Multiple paths
                matched_paths = [
                    p
                    for sub_path in paths
                    for p in cls.match_path(
                        sub_path,
                        match_regex=match_regex,
                        sort_paths=sort_paths,
                        **match_path_kwargs,
                    )
                ]
                if len(matched_paths) == 0:
                    raise FileNotFoundError(f"No paths could be matched with {paths}")
                if sync:
                    symbols = []
                    paths = symbol_dict()
                    for p in matched_paths:
                        s = cls.path_to_symbol(p, **path_to_symbol_kwargs)
                        symbols.append(s)
                        paths[s] = p
                elif len(symbols) != len(matched_paths):
                    raise ValueError("The number of symbols must be equal to the number of matched paths")
                else:
                    paths = symbol_dict({s: matched_paths[i] for i, s in enumerate(symbols)})
                if len(matched_paths) == 1 and single_path:
                    paths = matched_paths[0]
            else:
                raise TypeError(f"Path '{paths}' is not supported")
            if len(symbols) == 1 and single_symbol:
                symbols = symbols[0]

        return super(LocalData, cls).fetch(
            symbols,
            path=paths,
            **kwargs,
        )


CSVDataT = tp.TypeVar("CSVDataT", bound="CSVData")


class CSVData(LocalData):
    """Subclass of `Data` for data that can be fetched and updated using `pd.read_csv`."""

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        path: tp.Any = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        header: tp.Optional[tp.MaybeSequence[int]] = None,
        index_col: tp.Optional[int] = None,
        parse_dates: tp.Optional[bool] = None,
        squeeze: tp.Optional[bool] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_csv_kwargs,
    ) -> tp.Tuple[tp.SeriesFrame, dict]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load a CSV file.

        If `path` is None, uses `symbol` as the path to the CSV file.

        `skiprows` and `nrows` will be automatically calculated based on `start_row` and `end_row`.

        !!! note
            `start_row` and `end_row` must exclude header rows, while `end_row` must include the last row.

        Use `chunk_func` to select and concatenate chunks from `TextFileReader`. Gets called
        only if `iterator` or `chunksize` are set.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for other arguments.

        For defaults, see `custom.csv` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro._settings import settings

        csv_cfg = settings["data"]["custom"]["csv"]

        if start_row is None:
            start_row = csv_cfg["start_row"]
        if end_row is None:
            end_row = csv_cfg["end_row"]
        if header is None:
            header = csv_cfg["header"]
        if index_col is None:
            index_col = csv_cfg["index_col"]
        if parse_dates is None:
            parse_dates = csv_cfg["parse_dates"]
        if squeeze is None:
            squeeze = csv_cfg["squeeze"]
        read_csv_kwargs = merge_dicts(csv_cfg["read_csv_kwargs"], read_csv_kwargs)

        if path is None:
            path = symbol
        if isinstance(header, int):
            header = [header]
        header_rows = header[-1] + 1
        start_row += header_rows
        if end_row is not None:
            end_row += header_rows
        skiprows = range(header_rows, start_row)
        if end_row is not None:
            nrows = end_row - start_row + 1
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
        returned_kwargs = dict(last_row=start_row - header_rows + len(obj.index) - 1)
        return obj, returned_kwargs

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start_row"] = self.returned_kwargs[symbol]["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class HDFPathNotFoundError(Exception):
    """Gets raised if the path to an HDF file could not be found."""

    pass


class HDFKeyNotFoundError(Exception):
    """Gets raised if the key to an HDF object could not be found."""

    pass


HDFDataT = tp.TypeVar("HDFDataT", bound="HDFData")


class HDFData(LocalData):
    """Subclass of `Data` for data that can be fetched and updated using `pd.read_hdf`."""

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
        **kwargs,
    ) -> tp.List[Path]:
        """Override `LocalData.match_path` to return a list of HDF paths
        (path to file + key) matching a path."""
        path = Path(path)
        if path.exists():
            if path.is_dir():
                sub_paths = [p for p in path.iterdir() if p.is_file()]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
            else:
                with pd.HDFStore(str(path)) as store:
                    keys = [k[1:] for k in store.keys()]
                key_paths = [path / k for k in keys]
        else:
            try:
                file_path, key = cls.split_hdf_path(path)
                with pd.HDFStore(str(file_path)) as store:
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
                sub_paths = list([Path(p) for p in glob(str(path))])
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
                    sub_paths = list([Path(p) for p in glob(str(base_path))])
                    if key_path is not None:
                        sub_paths = [p / key_path for p in sub_paths]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
        if match_regex is not None:
            key_paths = [p for p in key_paths if re.match(match_regex, str(p))]
        if sort_paths:
            key_paths = sorted(key_paths)
        return key_paths

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        path: tp.Any = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_hdf_kwargs,
    ) -> tp.Tuple[tp.SeriesFrame, dict]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load an HDF object.

        If `path` is None, uses `symbol` as the path to the HDF file.

        !!! note
            `end_row` must include the last row.

        Use `chunk_func` to select and concatenate chunks from `TableIterator`. Gets called
        only if `iterator` or `chunksize` are set.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments.

        For defaults, see `custom.hdf` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro._settings import settings

        hdf_cfg = settings["data"]["custom"]["hdf"]

        if start_row is None:
            start_row = hdf_cfg["start_row"]
        if end_row is None:
            end_row = hdf_cfg["end_row"]
        read_hdf_kwargs = merge_dicts(hdf_cfg["read_hdf_kwargs"], read_hdf_kwargs)

        if path is None:
            path = symbol
        path = Path(path)
        file_path, key = cls.split_hdf_path(path)
        if end_row is not None:
            stop = end_row + 1
        else:
            stop = None

        obj = pd.read_hdf(file_path, key=key, start=start_row, stop=stop, **read_hdf_kwargs)
        if isinstance(obj, TableIterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        returned_kwargs = dict(last_row=start_row + len(obj.index) - 1)
        return obj, returned_kwargs

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start_row"] = self.returned_kwargs[symbol]["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


# ############# Remote ############# #


class RemoteData(Data):
    """Subclass of `Data` for remote data."""

    pass


class YFData(RemoteData):  # pragma: no cover
    """Subclass of `Data` for `yfinance`.

    See https://github.com/ranaroussi/yfinance."""

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        period: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        **history_kwargs,
    ) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Yahoo Finance.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            **history_kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.

        For defaults, see `custom.yf` in `vectorbtpro._settings.data`.

        Stocks are usually in the timezone "+0500" and cryptocurrencies in UTC.

        !!! warning
            Data coming from Yahoo is not the most stable data out there. Yahoo may manipulate data
            how they want, add noise, return missing data points (see volume in the example below), etc.
            It's only used in vectorbt for demonstration purposes.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("yfinance")
        import yfinance as yf

        from vectorbtpro._settings import settings

        yf_cfg = settings["data"]["custom"]["yf"]

        if period is None:
            period = yf_cfg["period"]
        if start is None:
            start = yf_cfg["start"]
        if end is None:
            end = yf_cfg["end"]
        history_kwargs = merge_dicts(yf_cfg["history_kwargs"], history_kwargs)

        # yfinance still uses mktime, which assumes that the passed date is in local time
        if start is not None:
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            end = to_tzaware_datetime(end, tz=get_local_tz())

        return yf.Ticker(symbol).history(period=period, start=start, end=end, **history_kwargs)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


class BinanceData(RemoteData):  # pragma: no cover
    """Subclass of `Data` for `python-binance`.

    See https://github.com/sammchardy/python-binance."""

    @classmethod
    def fetch(
        cls: tp.Type[BinanceDataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        client: tp.Optional["BinanceClientT"] = None,
        client_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BinanceDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to instantiate a Binance client.

        For defaults, see `custom.binance` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("binance")
        from binance.client import Client

        from vectorbtpro._settings import settings

        binance_cfg = settings["data"]["custom"]["binance"]

        if client is None:
            client = binance_cfg["client"]
        if client_kwargs is None:
            client_kwargs = {}
        has_client_kwargs = len(client_kwargs) > 0
        client_kwargs = merge_dicts(binance_cfg["client_kwargs"], client_kwargs)
        if client is None:
            client = Client(**client_kwargs)
        elif has_client_kwargs:
            raise ValueError("Cannot apply config after instantiation of the client")
        return super(BinanceData, cls).fetch(symbols, client=client, **kwargs)

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional["BinanceClientT"] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        interval: tp.Optional[str] = None,
        delay: tp.Optional[float] = None,
        limit: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Binance.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Client of type `binance.client.Client`.

                Must be provided.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            interval (str): Kline interval.

                See [Binance Constants](https://python-binance.readthedocs.io/en/latest/constants.html).
            delay (float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

        For defaults, see `custom.binance` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro._settings import settings

        binance_cfg = settings["data"]["custom"]["binance"]

        if client is None:
            client = binance_cfg["client"]
        if client is None:
            raise ValueError("Client must be provided")
        if start is None:
            start = binance_cfg["start"]
        if end is None:
            end = binance_cfg["end"]
        if interval is None:
            interval = binance_cfg["interval"]
        if delay is None:
            delay = binance_cfg["delay"]
        if limit is None:
            limit = binance_cfg["limit"]
        if show_progress is None:
            show_progress = binance_cfg["show_progress"]
        pbar_kwargs = merge_dicts(binance_cfg["pbar_kwargs"], pbar_kwargs)

        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1,
                startTime=0,
                endTime=None,
            )
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except Exception as e:
            next_start_ts = start_ts
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        def _ts_to_str(ts: tp.DatetimeLike) -> str:
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts))
                while True:
                    # Fetch the klines for the next interval
                    next_data = client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        startTime=next_start_ts,
                        endTime=end_ts,
                    )
                    if len(data) > 0:
                        next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                    else:
                        next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1][0]),
                        )
                    )
                    pbar.update(1)
                    next_start_ts = next_data[-1][0]
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            warnings.warn(traceback.format_exc(), stacklevel=2)
            warnings.warn(
                f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                "Use update() method to fetch missing data.",
                stacklevel=2,
            )

        # Convert data to a DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote volume",
                "Number of trades",
                "Taker base volume",
                "Taker quote volume",
                "Ignore",
            ],
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df["Close time"] = pd.to_datetime(df["Close time"], unit="ms", utc=True)
        df["Quote volume"] = df["Quote volume"].astype(float)
        df["Number of trades"] = df["Number of trades"].astype(int)
        df["Taker base volume"] = df["Taker base volume"].astype(float)
        df["Taker quote volume"] = df["Taker quote volume"].astype(float)
        del df["Ignore"]

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class CCXTData(RemoteData):  # pragma: no cover
    """Subclass of `Data` for `ccxt`.

    See https://github.com/ccxt/ccxt."""

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        exchange: tp.Optional[tp.Union[str, "ExchangeT"]] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        delay: tp.Optional[float] = None,
        limit: tp.Optional[int] = None,
        retries: tp.Optional[int] = None,
        exchange_config: tp.Optional[tp.KwargsLike] = None,
        fetch_params: tp.Optional[tp.KwargsLike] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from CCXT.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe supported by the exchange.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            exchange_config (dict): Keyword arguments passed to the exchange upon instantiation.

                Will raise an exception if exchange has been already instantiated.
            fetch_params (dict): Exchange-specific keyword arguments passed to `fetch_ohlcv`.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

        For defaults, see `custom.ccxt` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("ccxt")
        import ccxt

        from vectorbtpro._settings import settings

        ccxt_cfg = settings["data"]["custom"]["ccxt"]

        if exchange is None:
            exchange = ccxt_cfg["exchange"]
        if isinstance(exchange, str):
            exchange_name = exchange
        elif isinstance(exchange, ccxt.Exchange):
            exchange_name = exchange.__name__
        else:
            raise ValueError(f"Unknown exchange of type {type(exchange)}")
        if start is None:
            start = ccxt_cfg["exchanges"].get(exchange_name, {}).get("start", ccxt_cfg["start"])
        if end is None:
            end = ccxt_cfg["exchanges"].get(exchange_name, {}).get("end", ccxt_cfg["end"])
        if timeframe is None:
            timeframe = ccxt_cfg["exchanges"].get(exchange_name, {}).get("timeframe", ccxt_cfg["timeframe"])
        if delay is None:
            delay = ccxt_cfg["exchanges"].get(exchange_name, {}).get("delay", ccxt_cfg["delay"])
        if limit is None:
            limit = ccxt_cfg["exchanges"].get(exchange_name, {}).get("limit", ccxt_cfg["limit"])
        if retries is None:
            retries = ccxt_cfg["exchanges"].get(exchange_name, {}).get("retries", ccxt_cfg["retries"])
        if exchange_config is None:
            exchange_config = {}
        has_exchange_config = len(exchange_config) > 0
        exchange_config = merge_dicts(
            ccxt_cfg["exchange_config"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("exchange_config", {}),
            exchange_config,
        )
        fetch_params = merge_dicts(
            ccxt_cfg["fetch_params"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("fetch_params", {}),
            fetch_params,
        )
        if show_progress is None:
            show_progress = ccxt_cfg["exchanges"].get(exchange_name, {}).get("show_progress", ccxt_cfg["show_progress"])
        pbar_kwargs = merge_dicts(
            ccxt_cfg["pbar_kwargs"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("pbar_kwargs", {}),
            pbar_kwargs,
        )

        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange '{exchange}' not found in CCXT")
            exchange = getattr(ccxt, exchange)(exchange_config)
        else:
            if has_exchange_config:
                raise ValueError("Cannot apply config after instantiation of the exchange")
        if not exchange.has["fetchOHLCV"]:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")
        if exchange.has["fetchOHLCV"] == "emulated":
            warnings.warn("Using emulated OHLCV candles", stacklevel=2)

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        if i == retries - 1:
                            raise e
                    if delay is not None:
                        time.sleep(delay / 1000)

            return retry_method

        @_retry
        def _fetch(_since, _limit):
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=_since,
                limit=_limit,
                params=fetch_params,
            )

        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = _fetch(0, 1)
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except Exception as e:
            next_start_ts = start_ts
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        def _ts_to_str(ts):
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts))
                while True:
                    # Fetch the klines for the next interval
                    next_data = _fetch(next_start_ts, limit)
                    if len(data) > 0:
                        next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                    else:
                        next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1][0]),
                        )
                    )
                    pbar.update(1)
                    next_start_ts = next_data[-1][0]
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            warnings.warn(traceback.format_exc(), stacklevel=2)
            warnings.warn(
                f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                "Use update() method to fetch missing data.",
                stacklevel=2,
            )

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume"])
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


AlpacaDataT = tp.TypeVar("AlpacaDataT", bound="AlpacaData")


class AlpacaData(RemoteData):  # pragma: no cover
    """Subclass of `Data` for `alpaca-trade-api`.

    See https://github.com/alpacahq/alpaca-trade-api-python.

    Sign up for Alpaca API keys under https://app.alpaca.markets/signup.

    Contributed to vectorbt by @haxdds. Licensed under Apache 2.0 with Commons Clause license.
    Adapted to vectorbtpro by @polakowo."""

    @classmethod
    def fetch(
        cls: tp.Type[AlpacaDataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        client: tp.Optional["AlpacaClientT"] = None,
        client_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> AlpacaDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to instantiate an Alpaca client.

        For defaults, see `custom.alpaca` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("alpaca_trade_api")
        from alpaca_trade_api.rest import REST

        from vectorbtpro._settings import settings

        alpaca_cfg = settings["data"]["custom"]["alpaca"]

        if client is None:
            client = alpaca_cfg["client"]
        if client_kwargs is None:
            client_kwargs = {}
        has_client_kwargs = len(client_kwargs) > 0
        client_kwargs = merge_dicts(alpaca_cfg["client_kwargs"], client_kwargs)
        if client is None:
            client = REST(**client_kwargs)
        elif has_client_kwargs:
            raise ValueError("Cannot apply config after instantiation of the client")
        return super(AlpacaData, cls).fetch(symbols, client=client, **kwargs)

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional["AlpacaClientT"] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        adjustment: tp.Optional[str] = None,
        limit: tp.Optional[int] = None,
        exchange: tp.Optional[str] = None,
    ) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpaca.

        Args:
            symbol (str): Symbol.
            client (alpaca_trade_api.rest.REST): Client of type `alpaca_trade_api.rest.REST`.

                Must be provided.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe of data.

                Must be integer multiple of 'm' (minute), 'h' (hour) or 'd' (day). i.e. '15m'.
                See https://alpaca.markets/data.

                !!! note
                    Data from the latest 15 minutes is not available with a free data plan.
            adjustment (str): Specifies the corporate action adjustment for the stocks.

                Allowed are `raw`, `split`, `dividend`, and `all`.
            limit (int): The maximum number of returned items.
            exchange (str): For crypto symbols. Which exchange you wish to retrieve data from.

                Allowed are `FTX`, `ERSX`, and `CBSE`.

        For defaults, see `custom.alpaca` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("alpaca_trade_api")
        from alpaca_trade_api.rest import TimeFrameUnit, TimeFrame

        from vectorbtpro._settings import settings

        alpaca_cfg = settings["data"]["custom"]["alpaca"]

        if exchange is None:
            exchange = alpaca_cfg["exchange"]
        if isinstance(exchange, str):
            exchange_name = exchange
        else:
            raise ValueError(f"Unknown exchange of type {type(exchange)}")
        if client is None:
            client = alpaca_cfg["client"]
        if client is None:
            raise ValueError("Client must be provided")
        if start is None:
            start = alpaca_cfg["exchanges"].get(exchange_name, {}).get("start", alpaca_cfg["start"])
        if end is None:
            end = alpaca_cfg["exchanges"].get(exchange_name, {}).get("end", alpaca_cfg["end"])
        if timeframe is None:
            timeframe = alpaca_cfg["exchanges"].get(exchange_name, {}).get("timeframe", alpaca_cfg["timeframe"])
        if adjustment is None:
            adjustment = alpaca_cfg["exchanges"].get(exchange_name, {}).get("adjustment", alpaca_cfg["adjustment"])
        if limit is None:
            limit = alpaca_cfg["exchanges"].get(exchange_name, {}).get("limit", alpaca_cfg["limit"])

        _timeframe_units = {
            "d": TimeFrameUnit.Day,
            "h": TimeFrameUnit.Hour,
            "m": TimeFrameUnit.Minute,
        }

        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe '{timeframe}'")

        amount_str = timeframe[:-1]
        unit_str = timeframe[-1]

        if not amount_str.isnumeric() or unit_str not in _timeframe_units:
            raise ValueError(f"Invalid timeframe '{timeframe}'")

        amount = int(amount_str)
        unit = _timeframe_units[unit_str]

        _timeframe = TimeFrame(amount, unit)

        start_ts = to_tzaware_datetime(start, tz=get_utc_tz()).isoformat()
        end_ts = to_tzaware_datetime(end, tz=get_utc_tz()).isoformat()

        def _is_crypto_symbol(symbol):
            return len(symbol) == 6 and "USD" in symbol

        if _is_crypto_symbol(symbol):
            df = client.get_crypto_bars(
                symbol=symbol,
                timeframe=_timeframe,
                start=start_ts,
                end=end_ts,
                limit=limit,
                exchanges=[exchange],
            ).df
        else:
            df = client.get_bars(
                symbol=symbol,
                timeframe=_timeframe,
                start=start_ts,
                end=end_ts,
                adjustment=adjustment,
                limit=limit,
            ).df

        df.drop(["exchange"], axis=1, errors="ignore", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "Trade count",
                "vwap": "VWAP",
            },
            inplace=True,
        )

        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int)
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
