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
from pathlib import Path
from glob import glob

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data import nb
from vectorbtpro.data.base import Data, symbol_dict
from vectorbtpro.registries.jit_registry import jit_registry
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import get_utc_tz, get_local_tz, to_tzaware_datetime, datetime_to_ms
from vectorbtpro.utils.parsing import get_func_kwargs
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

LocalDataT = tp.TypeVar("LocalDataT", bound="LocalData")


def unfold_path(path: tp.PathLike) -> tp.List[Path]:
    """`unfold_path_func` that returns a list of files matching a path."""
    path = Path(path)
    if path.exists():
        if path.is_dir():
            return [p for p in path.iterdir() if p.is_file()]
        return [path]
    return list([Path(p) for p in glob(str(path))])


def path_to_symbol(path: tp.PathLike) -> str:
    """`path_to_symbol_func` that returns the stem of a path."""
    return Path(path).stem


class LocalData(Data):
    """`Data` for local data distributed over multiple files.

    Use either `symbols` or `path` to specify the path to one or multiple files.
    Allowed are paths in string or `pathlib.Path` format. Also allowed are string
    expressions accepted by `glob.glob`."""

    @classmethod
    def fetch(cls: tp.Type[LocalDataT],
              symbols: tp.Union[tp.Symbol, tp.Symbols] = None, *,
              path: tp.Any = None,
              parse_paths: bool = True,
              sort_paths: bool = True,
              unfold_path_func: tp.Callable = unfold_path,
              path_to_symbol_func: tp.Callable = path_to_symbol,
              **kwargs) -> LocalDataT:
        """Override `vectorbtpro.data.base.Data.fetch`.

        Set `parse_paths` to False to not parse paths and behave like a regular
        `vectorbtpro.data.base.Data` instance.

        Set `sort_paths` to False to disable sorting of found paths.

        Use `unfold_path_func` to unfold a path into multiple paths. Won't get applied
        if `path` is already an instance of `vectorbtpro.data.base.symbol_dict`.

        Use `path_to_symbol_func` to get the symbol from a path. Gets applied only when
        either `symbols` or `path` is None."""
        if parse_paths:
            sync = False
            if path is None:
                path = symbols
                sync = True
            elif symbols is None:
                sync = True
            if path is None:
                raise ValueError("At least symbols or path must be set")

            single_symbol = False
            if isinstance(symbols, (str, Path)):
                # Single symbol
                symbols = [symbols]
                single_symbol = True

            single_path = False
            if isinstance(path, (str, Path)):
                # Single path
                path = [path]
                single_path = True
                if sync:
                    single_symbol = True

            if isinstance(path, symbol_dict):
                # Dict of path per symbol
                if sync:
                    symbols = list(path.keys())
                elif len(symbols) != len(path):
                    raise ValueError("The number of symbols must match the number of paths")
            elif checks.is_iterable(path) or checks.is_sequence(path):
                # Multiple paths
                paths = [p for sub_path in path for p in unfold_path_func(sub_path)]
                if len(paths) == 0:
                    raise FileNotFoundError(f"No paths could be matched with {path}")
                if sort_paths:
                    paths = sorted(paths)
                if sync:
                    symbols = []
                    path = symbol_dict()
                    for p in paths:
                        s = path_to_symbol_func(p)
                        symbols.append(s)
                        path[s] = p
                elif len(symbols) != len(paths):
                    raise ValueError("The number of symbols must match the number of paths")
                else:
                    path = symbol_dict({s: paths[i] for i, s in enumerate(symbols)})
                if len(paths) == 1 and single_path:
                    path = paths[0]
            else:
                raise TypeError(f"Path '{path}' is not supported")
            if len(symbols) == 1 and single_symbol:
                symbols = symbols[0]

        return super(LocalData, cls).fetch(symbols, path=path, **kwargs)


CSVDataT = tp.TypeVar("CSVDataT", bound="CSVData")


class CSVData(LocalData):
    """`Data` for data that can be fetched and updated using `pd.read_csv`.

    Usage:
        * Generate three random time series, save them to the disk, and load using `CSVData`:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rand_data1 = vbt.RandomData.fetch(start='2020-01-01', end='2020-01-05')
        >>> rand_data2 = vbt.RandomData.fetch(start='2020-01-02', end='2020-01-05')
        >>> rand_data3 = vbt.RandomData.fetch(start='2020-01-03', end='2020-01-05')

        >>> rand_data1.get().to_csv('rand_data1.csv')
        >>> rand_data2.get().to_csv('rand_data2.csv')
        >>> rand_data3.get().to_csv('rand_data3.csv')

        >>> csv_data = vbt.CSVData.fetch('rand_data*.csv')
        >>> # same as vbt.CSVData.fetch(['rand_data1.csv', ...])
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> csv_data.get()
        symbol                     rand_data1.csv  rand_data2.csv  rand_data3.csv
        2019-12-31 23:00:00+00:00       99.632114             NaN             NaN
        2020-01-01 23:00:00+00:00       99.268772      101.372515             NaN
        2020-01-02 23:00:00+00:00       99.771735      101.609975       99.640045
        2020-01-03 23:00:00+00:00       98.452476      101.073658       99.466840
        2020-01-04 23:00:00+00:00       98.055655      101.846403       99.719464
        ```

        * Update one time series and update `CSVData`:

        ```pycon
        >>> rand_data3 = rand_data3.update(end='2020-01-07')
        >>> rand_data3.get().to_csv('rand_data3.csv')

        >>> csv_data = csv_data.update()  # loads only subset of data
        >>> csv_data.get()
        symbol                     rand_data1.csv  rand_data2.csv  rand_data3.csv
        2019-12-31 23:00:00+00:00       99.632114             NaN             NaN
        2020-01-01 23:00:00+00:00       99.268772      101.372515             NaN
        2020-01-02 23:00:00+00:00       99.771735      101.609975       99.640045
        2020-01-03 23:00:00+00:00       98.452476      101.073658       99.466840
        2020-01-04 23:00:00+00:00       98.055655      101.846403      100.212156
        2020-01-05 23:00:00+00:00             NaN             NaN      100.829512
        2020-01-06 23:00:00+00:00             NaN             NaN      100.617397
        ```
    """

    @classmethod
    def fetch_symbol(cls,
                     symbol: tp.Symbol,
                     path: tp.Any = None,
                     header: tp.MaybeSequence[int] = 0,
                     index_col: int = 0,
                     parse_dates: bool = True,
                     start_row: int = 0,
                     end_row: tp.Optional[int] = None,
                     squeeze: bool = True,
                     **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load a CSV file.

        If `path` is None, uses `symbol` as the path to the CSV file.

        `skiprows` and `nrows` will be automatically calculated based on `start_row` and `end_row`.

        !!! note
            `start_row` and `end_row` must exclude header rows, while `end_row` must include the last row.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for other arguments."""
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

        obj = pd.read_csv(
            path,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            skiprows=skiprows,
            nrows=nrows,
            squeeze=squeeze,
            **kwargs
        )
        if isinstance(obj, pd.Series) and obj.name == '0':
            obj.name = None
        returned_kwargs = dict(last_row=start_row - header_rows + len(obj.index) - 1)
        return obj, returned_kwargs

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start_row'] = self.returned_kwargs[symbol]['last_row']
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class HDFPathNotFoundError(Exception):
    """Gets raised if the path to an HDF file could not be found."""
    pass


class HDFKeyNotFoundError(Exception):
    """Gets raised if the key to an HDF object could not be found."""
    pass


def split_hdf_path(path: tp.PathLike,
                   key: tp.Optional[str] = None,
                   _full_path: tp.Optional[Path] = None) -> tp.Tuple[Path, tp.Optional[str]]:
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
    return split_hdf_path(new_path, new_key, _full_path=_full_path)


def hdf_unfold_path(path: tp.PathLike) -> tp.List[Path]:
    """`unfold_path_func` that returns a list of HDF paths (path to file + key) matching a path."""
    path = Path(path)
    if path.exists():
        if path.is_dir():
            sub_paths = [p for p in path.iterdir() if p.is_file()]
            return [p for sub_path in sub_paths for p in hdf_unfold_path(sub_path)]
        with pd.HDFStore(str(path)) as store:
            keys = [k[1:] for k in store.keys()]
        return [path / k for k in keys]
    try:
        file_path, key = split_hdf_path(path)
        with pd.HDFStore(str(file_path)) as store:
            keys = [k[1:] for k in store.keys()]
        if key is None:
            return [file_path / k for k in keys]
        if key in keys:
            return [file_path / key]
        matching_keys = []
        for k in keys:
            if k.startswith(key):
                matching_keys.append(k)
        if len(matching_keys) == 0:
            raise HDFKeyNotFoundError(f"No HDF keys could be matched with {key}")
        return [file_path / k for k in matching_keys]
    except HDFPathNotFoundError:
        pass
    sub_paths = list([Path(p) for p in glob(str(path))])
    return [p for sub_path in sub_paths for p in hdf_unfold_path(sub_path)]


HDFDataT = tp.TypeVar("HDFDataT", bound="HDFData")


class HDFData(LocalData):
    """`Data` for data that can be fetched and updated using `pd.read_hdf`.

    Usage:
        * Generate four random time series, save them to two HDF files, and load using `HDFData`:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rand_data1 = vbt.RandomData.fetch(start='2020-01-01', end='2020-01-05')
        >>> rand_data2 = vbt.RandomData.fetch(start='2020-01-02', end='2020-01-05')
        >>> rand_data3 = vbt.RandomData.fetch(start='2020-01-03', end='2020-01-05')
        >>> rand_data4 = vbt.RandomData.fetch(start='2020-01-04', end='2020-01-05')

        >>> rand_data1.get().to_hdf('rand_data1.h5', '/R1')
        >>> rand_data2.get().to_hdf('rand_data2.h5', '/R2')
        >>> rand_data3.get().to_hdf('rand_data2.h5', '/folder/R3')
        >>> rand_data4.get().to_hdf('rand_data2.h5', '/folder/R4')

        >>> hdf_data = vbt.HDFData.fetch('rand_data*.h5')
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> hdf_data.get()
        symbol                             R1          R2         R3          R4
        2019-12-31 23:00:00+00:00  101.351668         NaN        NaN         NaN
        2020-01-01 23:00:00+00:00  101.263132  100.999627        NaN         NaN
        2020-01-02 23:00:00+00:00  101.290664  103.431017  99.483751         NaN
        2020-01-03 23:00:00+00:00  103.178471  104.279458  99.207583   98.995509
        2020-01-04 23:00:00+00:00  103.493175  104.549302  99.784088  100.276962
        ```

        * Update one time series and update `HDFData`:

        ```pycon
        >>> rand_data4 = rand_data4.update(end='2020-01-07')
        >>> rand_data4.get().to_hdf('rand_data2.h5', '/folder/R4')

        >>> hdf_data = hdf_data.update()  # loads only subset of data
        >>> hdf_data.get()
        symbol                             R1          R2         R3         R4
        2019-12-31 23:00:00+00:00  101.351668         NaN        NaN        NaN
        2020-01-01 23:00:00+00:00  101.263132  100.999627        NaN        NaN
        2020-01-02 23:00:00+00:00  101.290664  103.431017  99.483751        NaN
        2020-01-03 23:00:00+00:00  103.178471  104.279458  99.207583  98.995509
        2020-01-04 23:00:00+00:00  103.493175  104.549302  99.784088  99.456149
        2020-01-05 23:00:00+00:00         NaN         NaN        NaN  96.833051
        2020-01-06 23:00:00+00:00         NaN         NaN        NaN  96.422318
        ```

        * Specify keys:

        ```pycon
        >>> vbt.HDFData.fetch('rand_data2.h5/R2').get()
        2020-01-01 23:00:00+00:00    100.999627
        2020-01-02 23:00:00+00:00    103.431017
        2020-01-03 23:00:00+00:00    104.279458
        2020-01-04 23:00:00+00:00    104.549302
        Freq: D, dtype: float6

        >>> vbt.HDFData.fetch('rand_data2.h5/folder').get()
        symbol                            R3         R4
        2020-01-02 23:00:00+00:00  99.483751        NaN
        2020-01-03 23:00:00+00:00  99.207583  98.995509
        2020-01-04 23:00:00+00:00  99.784088  99.456149
        2020-01-05 23:00:00+00:00        NaN  96.833051
        2020-01-06 23:00:00+00:00        NaN  96.422318
        ```
    """

    @classmethod
    def fetch(cls: tp.Type[HDFDataT],
              symbols: tp.Union[tp.Symbol, tp.Symbols] = None, *,
              unfold_path_func: tp.Callable = hdf_unfold_path,
              **kwargs) -> HDFDataT:
        """Override `vectorbtpro.data.base.LocalData.fetch` to parse paths and HDF keys."""
        return super(HDFData, cls).fetch(symbols, unfold_path_func=unfold_path_func, **kwargs)

    @classmethod
    def fetch_symbol(cls,
                     symbol: tp.Symbol,
                     path: tp.Any = None,
                     start_row: int = 0,
                     end_row: tp.Optional[int] = None,
                     **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load an HDF object.

        If `path` is None, uses `symbol` as the path to the HDF file.

        !!! note
            `end_row` must include the last row.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments."""
        if path is None:
            path = symbol
        path = Path(path)
        file_path, key = split_hdf_path(path)
        if end_row is not None:
            stop = end_row + 1
        else:
            stop = None

        obj = pd.read_hdf(
            file_path,
            key=key,
            start=start_row,
            stop=stop,
            **kwargs
        )
        returned_kwargs = dict(last_row=start_row + len(obj.index) - 1)
        return obj, returned_kwargs

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start_row'] = self.returned_kwargs[symbol]['last_row']
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class SyntheticData(Data):
    """`Data` for synthetically generated data.

    Exposes an abstract class method `SyntheticData.generate_symbol`.
    Everything else is taken care of."""

    @classmethod
    def generate_symbol(cls, symbol: tp.Symbol, index: tp.Index, **kwargs) -> tp.SeriesFrame:
        """Abstract method to generate data of a symbol."""
        raise NotImplementedError

    @classmethod
    def fetch_symbol(cls,
                     symbol: tp.Symbol,
                     start: tp.DatetimeLike = 0,
                     end: tp.DatetimeLike = 'now',
                     freq: tp.Union[None, str, pd.DateOffset] = None,
                     date_range_kwargs: tp.KwargsLike = None,
                     **kwargs) -> tp.SeriesFrame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to generate a symbol.

        Generates datetime index and passes it to `SyntheticData.generate_symbol` to fill
        the Series/DataFrame with generated data."""
        if date_range_kwargs is None:
            date_range_kwargs = {}
        index = pd.date_range(
            start=to_tzaware_datetime(start, tz=get_utc_tz()),
            end=to_tzaware_datetime(end, tz=get_utc_tz()),
            freq=freq,
            **date_range_kwargs
        )
        if len(index) == 0:
            raise ValueError("Date range is empty")
        return cls.generate_symbol(symbol, index, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class RandomData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_random_data_nb`.

    !!! note
        When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.

    Usage:
        * Generate random data:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rand_data = vbt.RandomData.fetch(
        ...     list(range(5)),
        ...     start='2010-01-01',
        ...     end='2020-01-01'
        ... )
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> rand_data.plot(showlegend=False)
        ```

        ![](/assets/images/RandomData.svg)
    """

    @classmethod
    def generate_symbol(cls,
                        symbol: tp.Symbol,
                        index: tp.Index,
                        num_paths: int = 1,
                        start_value: float = 100.,
                        mean: float = 0.,
                        std: float = 0.01,
                        seed: tp.Optional[int] = None,
                        jitted: tp.JittedOption = None) -> tp.SeriesFrame:
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
        """
        if seed is not None:
            set_seed(seed)

        func = jit_registry.resolve_option(nb.generate_random_data_nb, jitted)
        out = func((len(index), num_paths), start_value, mean, std)

        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name='path')
        return pd.DataFrame(out, index=index, columns=columns)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        _ = fetch_kwargs.pop('start_value', None)
        start_value = self.data[symbol].iloc[-2]
        fetch_kwargs['seed'] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)


class GBMData(RandomData):
    """`RandomData` for data generated using `vectorbtpro.data.nb.generate_gbm_data_nb`.

    !!! note
        When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.

    Usage:
        * Generate random data:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> gbm_data = vbt.GBMData.fetch(
        ...     list(range(5)),
        ...     start='2010-01-01',
        ...     end='2020-01-01'
        ... )
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> gbm_data.plot(showlegend=False)
        ```

        ![](/assets/images/GBMData.svg)
    """

    @classmethod
    def generate_symbol(cls,
                        symbol: tp.Symbol,
                        index: tp.Index,
                        num_paths: int = 1,
                        start_value: float = 100.,
                        mean: float = 0.,
                        std: float = 0.01,
                        dt: float = 1.,
                        seed: tp.Optional[int] = None,
                        jitted: tp.JittedOption = None) -> tp.SeriesFrame:
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
        """
        if seed is not None:
            set_seed(seed)

        func = jit_registry.resolve_option(nb.generate_gbm_data_nb, jitted)
        out = func((len(index), num_paths), start_value, mean, std, dt)

        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name='path')
        return pd.DataFrame(out, index=index, columns=columns)


class YFData(Data):  # pragma: no cover
    """`Data` for data coming from `yfinance`.

    Stocks are usually in the timezone "+0500" and cryptocurrencies in UTC.

    !!! warning
        Data coming from Yahoo is not the most stable data out there. Yahoo may manipulate data
        how they want, add noise, return missing data points (see volume in the example below), etc.
        It's only used in vectorbt for demonstration purposes.

    Usage:
        * Fetch the business day except the last 5 minutes of trading data:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> yf_data = vbt.YFData.fetch(
        ...     "TSLA",
        ...     start='2021-04-12 09:30:00 -0400',
        ...     end='2021-04-12 09:35:00 -0400',
        ...     interval='1m'
        ... )
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> yf_data.get()
                                         Open        High         Low       Close  \\
        Datetime
        2021-04-12 13:30:00+00:00  685.080017  685.679993  684.765015  685.679993
        2021-04-12 13:31:00+00:00  684.625000  686.500000  684.010010  685.500000
        2021-04-12 13:32:00+00:00  685.646790  686.820007  683.190002  686.455017
        2021-04-12 13:33:00+00:00  686.455017  687.000000  685.000000  685.565002
        2021-04-12 13:34:00+00:00  685.690002  686.400024  683.200012  683.715027

                                   Volume  Dividends  Stock Splits
        Datetime
        2021-04-12 13:30:00+00:00       0          0             0
        2021-04-12 13:31:00+00:00  152276          0             0
        2021-04-12 13:32:00+00:00  168363          0             0
        2021-04-12 13:33:00+00:00  129607          0             0
        2021-04-12 13:34:00+00:00  134620          0             0
        ```

        * Update with the missing 5 minutes:

        ```pycon
        >>> yf_data = yf_data.update(end='2021-04-12 09:40:00 -0400')
        >>> yf_data.get()
                                         Open        High         Low       Close  \\
        Datetime
        2021-04-12 13:30:00+00:00  685.080017  685.679993  684.765015  685.679993
        2021-04-12 13:31:00+00:00  684.625000  686.500000  684.010010  685.500000
        2021-04-12 13:32:00+00:00  685.646790  686.820007  683.190002  686.455017
        2021-04-12 13:33:00+00:00  686.455017  687.000000  685.000000  685.565002
        2021-04-12 13:34:00+00:00  685.690002  686.400024  683.200012  683.715027
        2021-04-12 13:35:00+00:00  683.604980  684.340027  682.760071  684.135010
        2021-04-12 13:36:00+00:00  684.130005  686.640015  683.333984  686.563904
        2021-04-12 13:37:00+00:00  686.530029  688.549988  686.000000  686.635010
        2021-04-12 13:38:00+00:00  686.593201  689.500000  686.409973  688.179993
        2021-04-12 13:39:00+00:00  688.500000  689.347595  687.710022  688.070007

                                   Volume  Dividends  Stock Splits
        Datetime
        2021-04-12 13:30:00+00:00       0          0             0
        2021-04-12 13:31:00+00:00  152276          0             0
        2021-04-12 13:32:00+00:00  168363          0             0
        2021-04-12 13:33:00+00:00  129607          0             0
        2021-04-12 13:34:00+00:00       0          0             0
        2021-04-12 13:35:00+00:00  110500          0             0
        2021-04-12 13:36:00+00:00  148384          0             0
        2021-04-12 13:37:00+00:00  243851          0             0
        2021-04-12 13:38:00+00:00  203569          0             0
        2021-04-12 13:39:00+00:00   93308          0             0
        ```
    """

    @classmethod
    def fetch_symbol(cls,
                     symbol: str,
                     period: str = 'max',
                     start: tp.Optional[tp.DatetimeLike] = None,
                     end: tp.Optional[tp.DatetimeLike] = None,
                     **kwargs) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Yahoo Finance.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            **kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('yfinance')
        import yfinance as yf

        # yfinance still uses mktime, which assumes that the passed date is in local time
        if start is not None:
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            end = to_tzaware_datetime(end, tz=get_local_tz())

        return yf.Ticker(symbol).history(period=period, start=start, end=end, **kwargs)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


class BinanceData(Data):  # pragma: no cover
    """`Data` for data coming from `python-binance`.

    Usage:
        * Fetch the 1-minute data of the last 2 hours, wait 1 minute:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> binance_data = vbt.BinanceData.fetch(
        ...     "BTCUSDT",
        ...     start='2 hours ago UTC',
        ...     end='now UTC',
        ...     interval='1m'
        ... )
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> binance_data.get()
        2021-05-02 14:47:20.478000+00:00 - 2021-05-02 16:47:00+00:00: : 1it [00:00,  3.42it/s]
                                       Open      High       Low     Close     Volume  \\
        Open time
        2021-05-02 14:48:00+00:00  56867.44  56913.57  56857.40  56913.56  28.709976
        2021-05-02 14:49:00+00:00  56913.56  56913.57  56845.94  56888.00  19.734841
        2021-05-02 14:50:00+00:00  56888.00  56947.32  56879.78  56934.71  23.150163
        ...                             ...       ...       ...       ...        ...
        2021-05-02 16:45:00+00:00  56664.13  56666.77  56641.11  56644.03  40.852719
        2021-05-02 16:46:00+00:00  56644.02  56663.43  56605.17  56605.18  27.573654
        2021-05-02 16:47:00+00:00  56605.18  56657.55  56605.17  56627.12   7.719933

                                                        Close time  Quote volume  \\
        Open time
        2021-05-02 14:48:00+00:00 2021-05-02 14:48:59.999000+00:00  1.633534e+06
        2021-05-02 14:49:00+00:00 2021-05-02 14:49:59.999000+00:00  1.122519e+06
        2021-05-02 14:50:00+00:00 2021-05-02 14:50:59.999000+00:00  1.317969e+06
        ...                                                    ...           ...
        2021-05-02 16:45:00+00:00 2021-05-02 16:45:59.999000+00:00  2.314579e+06
        2021-05-02 16:46:00+00:00 2021-05-02 16:46:59.999000+00:00  1.561548e+06
        2021-05-02 16:47:00+00:00 2021-05-02 16:47:59.999000+00:00  4.371848e+05

                                   Number of trades  Taker base volume  \\
        Open time
        2021-05-02 14:48:00+00:00               991          13.771152
        2021-05-02 14:49:00+00:00               816           5.981942
        2021-05-02 14:50:00+00:00              1086          10.813757
        ...                                     ...                ...
        2021-05-02 16:45:00+00:00              1006          18.106933
        2021-05-02 16:46:00+00:00               916          14.869411
        2021-05-02 16:47:00+00:00               353           3.903321

                                   Taker quote volume
        Open time
        2021-05-02 14:48:00+00:00        7.835391e+05
        2021-05-02 14:49:00+00:00        3.402170e+05
        2021-05-02 14:50:00+00:00        6.156418e+05
        ...                                       ...
        2021-05-02 16:45:00+00:00        1.025892e+06
        2021-05-02 16:46:00+00:00        8.421173e+05
        2021-05-02 16:47:00+00:00        2.210323e+05

        [120 rows x 10 columns]
        ```

        * Update data:

        ```pycon
        >>> import time
        >>> time.sleep(60)

        >>> binance_data = binance_data.update()
        >>> binance_data.get()
                                       Open      High       Low     Close     Volume  \\
        Open time
        2021-05-02 14:48:00+00:00  56867.44  56913.57  56857.40  56913.56  28.709976
        2021-05-02 14:49:00+00:00  56913.56  56913.57  56845.94  56888.00  19.734841
        2021-05-02 14:50:00+00:00  56888.00  56947.32  56879.78  56934.71  23.150163
        ...                             ...       ...       ...       ...        ...
        2021-05-02 16:46:00+00:00  56644.02  56663.43  56605.17  56605.18  27.573654
        2021-05-02 16:47:00+00:00  56605.18  56657.55  56605.17  56625.76  14.615437
        2021-05-02 16:48:00+00:00  56625.75  56643.60  56614.32  56623.01   5.895843

                                                        Close time  Quote volume  \\
        Open time
        2021-05-02 14:48:00+00:00 2021-05-02 14:48:59.999000+00:00  1.633534e+06
        2021-05-02 14:49:00+00:00 2021-05-02 14:49:59.999000+00:00  1.122519e+06
        2021-05-02 14:50:00+00:00 2021-05-02 14:50:59.999000+00:00  1.317969e+06
        ...                                                    ...           ...
        2021-05-02 16:46:00+00:00 2021-05-02 16:46:59.999000+00:00  1.561548e+06
        2021-05-02 16:47:00+00:00 2021-05-02 16:47:59.999000+00:00  8.276017e+05
        2021-05-02 16:48:00+00:00 2021-05-02 16:48:59.999000+00:00  3.338702e+05

                                   Number of trades  Taker base volume  \\
        Open time
        2021-05-02 14:48:00+00:00               991          13.771152
        2021-05-02 14:49:00+00:00               816           5.981942
        2021-05-02 14:50:00+00:00              1086          10.813757
        ...                                     ...                ...
        2021-05-02 16:46:00+00:00               916          14.869411
        2021-05-02 16:47:00+00:00               912           7.778489
        2021-05-02 16:48:00+00:00               308           2.358130

                                   Taker quote volume
        Open time
        2021-05-02 14:48:00+00:00        7.835391e+05
        2021-05-02 14:49:00+00:00        3.402170e+05
        2021-05-02 14:50:00+00:00        6.156418e+05
        ...                                       ...
        2021-05-02 16:46:00+00:00        8.421173e+05
        2021-05-02 16:47:00+00:00        4.404362e+05
        2021-05-02 16:48:00+00:00        1.335474e+05

        [121 rows x 10 columns]
        ```
    """

    @classmethod
    def fetch(cls: tp.Type[BinanceDataT],
              symbols: tp.Union[tp.Symbol, tp.Symbols] = None, *,
              client: tp.Optional["BinanceClientT"] = None,
              **kwargs) -> BinanceDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to instantiate a Binance client."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('binance')
        from binance.client import Client

        from vectorbtpro._settings import settings
        binance_cfg = settings['data']['custom']['binance']

        client_kwargs = dict()
        for k in get_func_kwargs(Client.__init__):
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        client_kwargs = merge_dicts(binance_cfg, client_kwargs)
        if client is None:
            client = Client(**client_kwargs)
        return super(BinanceData, cls).fetch(symbols, client=client, **kwargs)

    @classmethod
    def fetch_symbol(cls,
                     symbol: str,
                     client: tp.Optional["BinanceClientT"] = None,
                     interval: str = '1d',
                     start: tp.DatetimeLike = 0,
                     end: tp.DatetimeLike = 'now UTC',
                     delay: tp.Optional[float] = 500,
                     limit: int = 500,
                     show_progress: bool = True,
                     pbar_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Binance.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Client of type `binance.client.Client`.
            interval (str): Kline interval.

                See `binance.enums`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

        For defaults, see `custom.binance` in `vectorbtpro._settings.data`.
        """
        if client is None:
            raise ValueError("client must be provided")

        if pbar_kwargs is None:
            pbar_kwargs = {}
        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1,
                startTime=0,
                endTime=None
            )
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
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
                        endTime=end_ts
                    )
                    if len(data) > 0:
                        next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                    else:
                        next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    pbar.set_description("{} - {}".format(
                        _ts_to_str(start_ts),
                        _ts_to_str(next_data[-1][0])
                    ))
                    pbar.update(1)
                    next_start_ts = next_data[-1][0]
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            warnings.warn(traceback.format_exc())
            warnings.warn(f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                          f"Use update() method to fetch missing data.", stacklevel=2)

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=[
            'Open time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Close time',
            'Quote volume',
            'Number of trades',
            'Taker base volume',
            'Taker quote volume',
            'Ignore'
        ])
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        del df['Open time']
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms', utc=True)
        df['Quote volume'] = df['Quote volume'].astype(float)
        df['Number of trades'] = df['Number of trades'].astype(int)
        df['Taker base volume'] = df['Taker base volume'].astype(float)
        df['Taker quote volume'] = df['Taker quote volume'].astype(float)
        del df['Ignore']

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class CCXTData(Data):  # pragma: no cover
    """`Data` for data coming from `ccxt`.

    Usage:
        * Fetch the 1-minute data of the last 2 hours, wait 1 minute:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> ccxt_data = vbt.CCXTData.fetch(
        ...     "BTC/USDT",
        ...     start='2 hours ago UTC',
        ...     end='now UTC',
        ...     timeframe='1m'
        ... )
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> ccxt_data.get()
        2021-05-02 14:50:26.305000+00:00 - 2021-05-02 16:50:00+00:00: : 1it [00:00,  1.96it/s]
                                       Open      High       Low     Close     Volume
        Open time
        2021-05-02 14:51:00+00:00  56934.70  56964.59  56910.00  56948.99  22.158319
        2021-05-02 14:52:00+00:00  56948.99  56999.00  56940.04  56977.62  46.958464
        2021-05-02 14:53:00+00:00  56977.61  56987.09  56882.98  56885.42  27.752200
        ...                             ...       ...       ...       ...        ...
        2021-05-02 16:48:00+00:00  56625.75  56643.60  56595.47  56596.01  15.452510
        2021-05-02 16:49:00+00:00  56596.00  56664.14  56596.00  56640.35  12.777475
        2021-05-02 16:50:00+00:00  56640.35  56675.82  56640.35  56670.65   6.882321

        [120 rows x 5 columns]
        ```

        * Update data:

        ```pycon
        >>> import time
        >>> time.sleep(60)

        >>> ccxt_data = ccxt_data.update()
        >>> ccxt_data.get()
                                       Open      High       Low     Close     Volume
        Open time
        2021-05-02 14:51:00+00:00  56934.70  56964.59  56910.00  56948.99  22.158319
        2021-05-02 14:52:00+00:00  56948.99  56999.00  56940.04  56977.62  46.958464
        2021-05-02 14:53:00+00:00  56977.61  56987.09  56882.98  56885.42  27.752200
        ...                             ...       ...       ...       ...        ...
        2021-05-02 16:49:00+00:00  56596.00  56664.14  56596.00  56640.35  12.777475
        2021-05-02 16:50:00+00:00  56640.35  56689.99  56640.35  56678.33  14.610231
        2021-05-02 16:51:00+00:00  56678.33  56688.99  56636.89  56653.42  11.647158

        [121 rows x 5 columns]
        ```
    """

    @classmethod
    def fetch_symbol(cls,
                     symbol: str,
                     exchange: tp.Union[str, "ExchangeT"] = 'binance',
                     config: tp.Optional[dict] = None,
                     timeframe: str = '1d',
                     start: tp.DatetimeLike = 0,
                     end: tp.DatetimeLike = 'now UTC',
                     delay: tp.Optional[float] = None,
                     limit: tp.Optional[int] = 500,
                     retries: int = 3,
                     show_progress: bool = True,
                     params: tp.Optional[dict] = None,
                     pbar_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from CCXT.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            config (dict): Config passed to the exchange upon instantiation.

                Will raise an exception if exchange has been already instantiated.
            timeframe (str): Timeframe supported by the exchange.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
            params (dict): Exchange-specific key-value parameters.

        For defaults, see `custom.ccxt` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('ccxt')
        import ccxt

        from vectorbtpro._settings import settings
        ccxt_cfg = settings['data']['custom']['ccxt']

        if config is None:
            config = {}
        if pbar_kwargs is None:
            pbar_kwargs = {}
        if params is None:
            params = {}
        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange {exchange} not found")
            # Resolve config
            default_config = {}
            for k, v in ccxt_cfg.items():
                # Get general (not per exchange) settings
                if k in ccxt.exchanges:
                    continue
                default_config[k] = v
            if exchange in ccxt_cfg:
                default_config = merge_dicts(default_config, ccxt_cfg[exchange])
            config = merge_dicts(default_config, config)
            exchange = getattr(ccxt, exchange)(config)
        else:
            if len(config) > 0:
                raise ValueError("Cannot apply config after instantiation of the exchange")
        if not exchange.has['fetchOHLCV']:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")
        if exchange.has['fetchOHLCV'] == 'emulated':
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
                params=params
            )

        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = _fetch(0, 1)
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
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
                    pbar.set_description("{} - {}".format(
                        _ts_to_str(start_ts),
                        _ts_to_str(next_data[-1][0])
                    ))
                    pbar.update(1)
                    next_start_ts = next_data[-1][0]
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            warnings.warn(traceback.format_exc())
            warnings.warn(f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                          f"Use update() method to fetch missing data.", stacklevel=2)

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=[
            'Open time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ])
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        del df['Open time']
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


AlpacaDataT = tp.TypeVar("AlpacaDataT", bound="AlpacaData")


class AlpacaData(Data):  # pragma: no cover
    """`Data` for data coming from `alpaca-trade-api`.

    Sign up for Alpaca API keys under https://app.alpaca.markets/signup.

    Contributed to vectorbt by @haxdds. Licensed under Apache 2.0 with Commons Clause license.
    Adapted to vectorbtpro by @polakowo.

    Usage:
        * Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.settings['data']['custom']['alpaca']['key_id'] = "{API Key ID}"
        >>> vbt.settings['data']['custom']['alpaca']['secret_key'] = "{Secret Key}"

        >>> alpaca_data = vbt.AlpacaData.fetch(
        ...     "AAPL",
        ...     start='2 hours ago UTC',
        ...     end='15 minutes ago UTC',
        ...     interval='1m'
        ... )
        >>> alpaca_data.get()
                                    Open      High       Low     Close      Volume
        timestamp
        2021-12-27 14:04:00+00:00  177.0500  177.0500  177.0500  177.0500    1967
        2021-12-27 14:05:00+00:00  177.0500  177.0500  177.0300  177.0500    3218
        2021-12-27 14:06:00+00:00  177.0400  177.0400  177.0400  177.0400     873
        ...                             ...       ...       ...       ...     ...
        2021-12-27 15:46:00+00:00  177.9500  178.0000  177.8289  177.8850  162778
        2021-12-27 15:47:00+00:00  177.8810  177.9600  177.8400  177.9515  123284
        2021-12-27 15:48:00+00:00  177.9600  178.0500  177.9600  178.0100  159700

        [105 rows x 5 columns]

        >>> import time
        >>> time.sleep(60)

        >>> alpaca_data = alpaca_data.update()
        >>> alpaca_data.get()
                                    Open      High       Low     Close      Volume
        timestamp
        2021-12-27 14:04:00+00:00  177.0500  177.0500  177.0500  177.0500    1967
        2021-12-27 14:05:00+00:00  177.0500  177.0500  177.0300  177.0500    3218
        2021-12-27 14:06:00+00:00  177.0400  177.0400  177.0400  177.0400     873
        ...                             ...       ...       ...       ...     ...
        2021-12-27 15:47:00+00:00  177.8810  177.9600  177.8400  177.9515  123284
        2021-12-27 15:48:00+00:00  177.9600  178.0500  177.9600  178.0100  159700
        2021-12-27 15:49:00+00:00  178.0100  178.0700  177.9700  178.0650  185037

        [106 rows x 5 columns]
        ```
    """

    @classmethod
    def fetch(cls: tp.Type[AlpacaDataT],
              symbols: tp.Union[tp.Symbol, tp.Symbols] = None, *,
              client: tp.Optional["AlpacaClientT"] = None,
              **kwargs) -> AlpacaDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to instantiate an Alpaca client."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('alpaca_trade_api')
        from alpaca_trade_api.rest import REST

        from vectorbtpro._settings import settings
        alpaca_cfg = settings['data']['custom']['alpaca']

        client_kwargs = dict()
        for k in get_func_kwargs(REST.__init__):
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        client_kwargs = merge_dicts(alpaca_cfg, client_kwargs)
        if client is None:
            client = REST(**client_kwargs)
        return super(AlpacaData, cls).fetch(symbols, client=client, **kwargs)

    @classmethod
    def fetch_symbol(cls,
                     symbol: str,
                     client: tp.Optional["AlpacaClientT"] = None,
                     timeframe: str = '1d',
                     start: tp.DatetimeLike = 0,
                     end: tp.DatetimeLike = 'now UTC',
                     adjustment: tp.Optional[str] = 'all',
                     limit: int = 500,
                     exchange: tp.Optional[str] = 'CBSE',
                     **kwargs) -> tp.Frame:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpaca.

        Args:
            symbol (str): Symbol.
            client (alpaca_trade_api.rest.REST): Client of type `alpaca_trade_api.rest.REST`.
            timeframe (str): Timeframe of data.

                Must be integer multiple of 'm' (minute), 'h' (hour) or 'd' (day). i.e. '15m'.
                See https://alpaca.markets/data.

                !!! note
                    Data from the latest 15 minutes is not available with a free data plan.

            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            adjustment (str): Specifies the corporate action adjustment for the stocks.

                Allowed are `raw`, `split`, `dividend`, and `all`.
            limit (int): The maximum number of returned items.
            exchange (str): For crypto symbols. Which exchange you wish to retrieve data from.

                Allowed are `FTX`, `ERSX`, and `CBSE`.

        For defaults, see `custom.alpaca` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('alpaca_trade_api')
        from alpaca_trade_api.rest import TimeFrameUnit, TimeFrame

        _timeframe_units = {'d': TimeFrameUnit.Day, 'h': TimeFrameUnit.Hour, 'm': TimeFrameUnit.Minute}

        if len(timeframe) < 2:
            raise ValueError("invalid timeframe")

        amount_str = timeframe[:-1]
        unit_str = timeframe[-1]

        if not amount_str.isnumeric() or unit_str not in _timeframe_units:
            raise ValueError("invalid timeframe")

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
                exchanges=exchange
            ).df
        else:
            df = client.get_bars(
                symbol=symbol,
                timeframe=_timeframe,
                start=start_ts,
                end=end_ts,
                adjustment=adjustment,
                limit=limit
            ).df

        # filter for OHLCV
        # remove extra columns
        df.drop(['trade_count', 'vwap'], axis=1, errors='ignore', inplace=True)

        # capitalize
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'exchange': 'Exchange'
        }, inplace=True)

        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start'] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
