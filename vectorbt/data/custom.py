# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Custom data classes that subclass `vectorbt.data.base.Data`.

!!! note
    Use absolute start and end dates instead of relative ones when fetching multiple
    symbols of data: some symbols may take a considerable amount of time to fetch
    such that they may shift the time period for the symbols coming next.

    This happens when relative dates are parsed in `vectorbt.data.base.Data.fetch_symbol`
    instead of parsing them once and for all symbols in `vectorbt.data.base.Data.fetch`."""

import time
import traceback
import warnings
from functools import wraps

import pandas as pd

from vectorbt import _typing as tp
from vectorbt.data import nb
from vectorbt.data.base import Data
from vectorbt.jit_registry import jit_registry
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime_ import get_utc_tz, get_local_tz, to_tzaware_datetime, datetime_to_ms
from vectorbt.utils.parsing import get_func_kwargs
from vectorbt.utils.pbar import get_pbar
from vectorbt.utils.random_ import set_seed

try:
    from binance.client import Client as ClientT
except ImportError:
    ClientT = tp.Any
try:
    from ccxt.base.exchange import Exchange as ExchangeT
except ImportError:
    ExchangeT = tp.Any

CSVDataT = tp.TypeVar("CSVDatat", bound="CSVData")


class CSVData(Data):
    """`Data` for data that can be fetched and updated using `pd.read_csv`.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> rand_data = vbt.RandomData.fetch(start='5 seconds ago', freq='1s')
    >>> rand_data.get().to_csv('rand_data.csv')

    >>> csv_data = vbt.CSVData.fetch('rand_data.csv')
    >>> csv_data.get()
    2021-10-30 11:19:32.168721+00:00    101.223837
    2021-10-30 11:19:33.168721+00:00    101.580351
    2021-10-30 11:19:34.168721+00:00    101.409540
    2021-10-30 11:19:35.168721+00:00    101.198202
    2021-10-30 11:19:36.168721+00:00    102.308458
    2021-10-30 11:19:37.168721+00:00    102.692657
    Freq: S, dtype: float64

    >>> import time
    >>> time.sleep(2)

    >>> rand_data = rand_data.update()
    >>> rand_data.get().to_csv('rand_data.csv')  # saves all data

    >>> csv_data = csv_data.update()  # loads only subset of data
    >>> csv_data.get()
    2021-10-30 11:19:32.168721+00:00    101.223837
    2021-10-30 11:19:33.168721+00:00    101.580351
    2021-10-30 11:19:34.168721+00:00    101.409540
    2021-10-30 11:19:35.168721+00:00    101.198202
    2021-10-30 11:19:36.168721+00:00    102.308458
    2021-10-30 11:19:37.168721+00:00    100.941811  << updated last
    2021-10-30 11:19:38.168721+00:00    100.935500  << added new
    2021-10-30 11:19:39.168721+00:00    100.618909  << added new
    Freq: S, dtype: float64
    ```"""

    @classmethod
    def fetch_symbol(cls,
                     symbol: tp.Symbol,
                     path: tp.Optional[tp.Any] = None,
                     header: tp.MaybeSequence[int] = 0,
                     index_col: int = 0,
                     parse_dates: bool = True,
                     start_row: int = 0,
                     end_row: tp.Optional[int] = None,
                     squeeze: bool = True,
                     **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.Kwargs]:
        """Fetch a symbol.

        If `path` is None, uses `symbol` as path.

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

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.Kwargs]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs['start_row'] = self.returned_kwargs[symbol]['last_row']
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class HDFData(Data):
    """`Data` for data that can be fetched and updated using `pd.read_hdf`.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> rand_data = vbt.RandomData.fetch(start='5 seconds ago', freq='1s')
    >>> rand_data.get().to_hdf('rand_data.h5', 's')

    >>> h5_data = vbt.HDFData.fetch('s', path='rand_data.h5')
    >>> h5_data.get()
    2021-10-30 17:53:14.243189+00:00    100.119015
    2021-10-30 17:53:15.243189+00:00    100.798207
    2021-10-30 17:53:16.243189+00:00    101.928613
    2021-10-30 17:53:17.243189+00:00    102.828592
    2021-10-30 17:53:18.243189+00:00    104.505259
    2021-10-30 17:53:19.243189+00:00    106.384834
    Freq: S, dtype: float64

    >>> import time
    >>> time.sleep(2)

    >>> rand_data = rand_data.update()
    >>> rand_data.get().to_hdf('rand_data.h5', 's')  # saves all data

    >>> h5_data = h5_data.update()  # loads only subset of data
    >>> h5_data.get()
    2021-10-30 17:53:14.243189+00:00    100.119015
    2021-10-30 17:53:15.243189+00:00    100.798207
    2021-10-30 17:53:16.243189+00:00    101.928613
    2021-10-30 17:53:17.243189+00:00    102.828592
    2021-10-30 17:53:18.243189+00:00    104.505259
    2021-10-30 17:53:19.243189+00:00    104.143708
    2021-10-30 17:53:20.243189+00:00    104.661706
    2021-10-30 17:53:21.243189+00:00    105.294653
    Freq: S, dtype: float64
    ```"""

    @classmethod
    def fetch_symbol(cls,
                     symbol: tp.Symbol,
                     path: tp.Optional[tp.Any] = None,
                     start_row: int = 0,
                     end_row: tp.Optional[int] = None,
                     **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.Kwargs]:
        """Fetch a symbol.

        If `path` is None, uses `symbol` as path to an HDF file with a single pandas object.

        !!! note
            `end_row` must include the last row.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments."""
        if path is None:
            path = symbol
            key = None
        else:
            key = symbol
        if end_row is not None:
            stop = end_row + 1
        else:
            stop = None
        obj = pd.read_hdf(
            path,
            key=key,
            start=start_row,
            stop=stop,
            **kwargs
        )
        returned_kwargs = dict(last_row=start_row + len(obj.index) - 1)
        return obj, returned_kwargs

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.Kwargs]:
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
        """Fetch a symbol.

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
    """`SyntheticData` for data generated using `vectorbt.data.nb.generate_random_data_nb`.

    !!! note
        When setting a seed, remember to pass a seed per symbol using `vectorbt.data.base.symbol_dict`.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> rand_data = vbt.RandomData.fetch(
    ...     list(range(10)),
    ...     start='2010-01-01',
    ...     end='2020-01-01'
    ... )

    >>> rand_data.plot(showlegend=False)
    ```

    ![](/docs/img/RandomData.svg)
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
            jitted (any): See `vectorbt.utils.jitting.resolve_jitted_option`.
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
    """`RandomData` for data generated using `vectorbt.data.nb.generate_gbm_data_nb`.

    !!! note
        When setting a seed, remember to pass a seed per symbol using `vectorbt.data.base.symbol_dict`.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> gbm_data = vbt.GBMData.fetch(
    ...     list(range(10)),
    ...     start='2010-01-01',
    ...     end='2020-01-01'
    ... )

    >>> gbm_data.plot(showlegend=False)
    ```

    ![](/docs/img/GBMData.svg)
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
            jitted (any): See `vectorbt.utils.jitting.resolve_jitted_option`.
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

    ## Example

    Fetch the business day except the last 5 minutes of trading data, and then update with the missing 5 minutes:

    ```python-repl
    >>> import vectorbt as vbt

    >>> yf_data = vbt.YFData.fetch(
    ...     "TSLA",
    ...     start='2021-04-12 09:30:00 -0400',
    ...     end='2021-04-12 09:35:00 -0400',
    ...     interval='1m'
    ... )
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
        """Fetch a symbol.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            **kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.
        """
        from vectorbt.opt_packages import assert_can_import
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

    ## Example

    Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

    ```python-repl
    >>> import vectorbt as vbt

    >>> binance_data = vbt.BinanceData.fetch(
    ...     "BTCUSDT",
    ...     start='2 hours ago UTC',
    ...     end='now UTC',
    ...     interval='1m'
    ... )
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
    ```"""

    @classmethod
    def fetch(cls: tp.Type[BinanceDataT],
              symbols: tp.Sequence[str],
              client: tp.Optional["ClientT"] = None,
              **kwargs) -> BinanceDataT:
        """Override `vectorbt.data.base.Data.fetch` to instantiate a Binance client."""
        from vectorbt.opt_packages import assert_can_import
        assert_can_import('binance')
        from binance.client import Client

        from vectorbt._settings import settings
        binance_cfg = settings['data']['custom']['binance']

        client_kwargs = dict()
        for k in get_func_kwargs(Client):
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        client_kwargs = merge_dicts(binance_cfg, client_kwargs)
        if client is None:
            client = Client(**client_kwargs)
        return super(BinanceData, cls).fetch(symbols, client=client, **kwargs)

    @classmethod
    def fetch_symbol(cls,
                     symbol: str,
                     client: tp.Optional["ClientT"] = None,
                     interval: str = '1d',
                     start: tp.DatetimeLike = 0,
                     end: tp.DatetimeLike = 'now UTC',
                     delay: tp.Optional[float] = 500,
                     limit: int = 500,
                     show_progress: bool = True,
                     pbar_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Fetch a symbol.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Binance client of type `binance.client.Client`.
            interval (str): Kline interval.

                See `binance.enums`.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbt.utils.pbar.get_pbar`.

        For defaults, see `data.custom.binance` in `vectorbt._settings.settings`.
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

    ## Example

    Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

    ```python-repl
    >>> import vectorbt as vbt

    >>> ccxt_data = vbt.CCXTData.fetch(
    ...     "BTC/USDT",
    ...     start='2 hours ago UTC',
    ...     end='now UTC',
    ...     timeframe='1m'
    ... )
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
    ```"""

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
        """Fetch a symbol.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            config (dict): Config passed to the exchange upon instantiation.

                Will raise an exception if exchange has been already instantiated.
            timeframe (str): Timeframe supported by the exchange.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbt.utils.pbar.get_pbar`.
            params (dict): Exchange-specific key-value parameters.

        For defaults, see `custom.data.ccxt` in `vectorbt._settings.settings`.
        """
        from vectorbt.opt_packages import assert_can_import
        assert_can_import('ccxt')
        import ccxt

        from vectorbt._settings import settings
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
