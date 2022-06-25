---
title: Features
hide:
  - navigation
---

# Features :zap:

On top of the [features](https://vectorbt.dev/getting-started/features/) offered by the community 
version of vectorbt, vectorbt PRO implements the following major enhancements:

## Performance

### Parallelization with Numba

- [x] Most Numba-compiled functions were rewritten to process columns in parallel, which can be 
enabled by a single command.

```pycon title="Benchmark the rolling mean without and with parallelization"
>>> import vectorbtpro as vbt
>>> import pandas as pd
>>> import numpy as np
>>> from numba import njit

>>> df = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit df.rolling(10).mean()  # (1)!
45.6 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit df.vbt.rolling_mean(10)  # (2)!
5.33 ms ± 302 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit df.vbt.rolling_mean(10, jitted=dict(parallel=True))  # (3)!
1.82 ms ± 5.21 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

1. Using Pandas
2. Using Numba without parallelization
3. Using Numba with parallelization

### Chunking

- [x] Innovative chunking mechanism that takes a specification of how arguments should be chunked, 
automatically splits them, passes each chunk to the function, and merges back the results. This way, 
you can run any function in a distributed manner! Additionally, vectorbt PRO implements a central 
registry and provides the chunking specification for all arguments of most Numba-compiled functions. 
Chunking can be enabled by a single command. No more out-of-memory errors! :tada:

```pycon title="Split hyperparameters into chunks and backtest them"
>>> price = vbt.YFData.fetch(['BTC-USD', 'ETH-USD']).get('Close')

>>> @vbt.chunked(
...     size=vbt.LenSizer(arg_query='fast_windows'),
...     arg_take_spec=dict(
...         price=None,
...         fast_windows=vbt.ChunkSlicer(),
...         slow_windows=vbt.ChunkSlicer()
...     ),
...     merge_func=lambda x: pd.concat(x).vbt.sort_index(),
...     show_progress=True
... )
... def get_total_return(price, fast_windows, slow_windows):
...     fast_ma = vbt.MA.run(price, fast_windows, short_name='fast')
...     slow_ma = vbt.MA.run(price, slow_windows, short_name='slow')
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     pf = vbt.Portfolio.from_signals(price, entries, exits)
...     return pf.total_return

>>> get_total_return(price, [10, 10, 10], [20, 30, 50])
```

[=100% "Chunk 3/3"]{: .candystripe}

```pycon
fast_window  slow_window  symbol 
10           20           BTC-USD     4.161972
                          ETH-USD    20.806896
             30           BTC-USD     5.797877
                          ETH-USD    12.573001
             50           BTC-USD     3.218610
                          ETH-USD     7.651010
Name: total_return, dtype: float64
```

### Multithreading

- [x] Integration of the [Dask](https://dask.org/) backend for running multiple chunks across 
multiple threads. Best suited for accelerating functions that release GIL, such as Numba and C 
functions. Cuts down execution time on Apple M1 by 3-4x, and even more depending on the number 
of cores. Dask + Numba = :muscle:

```pycon title="Benchmark 10 million orders without and with multithreading"
>>> big_size = np.full((10000, 1000), 0)

>>> %timeit vbt.Portfolio.from_orders(close=1, size=big_size)
856 ms ± 31.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit vbt.Portfolio.from_orders(close=1, size=big_size, chunked='dask')
262 ms ± 9.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

### Multiprocessing

- [x] Integration of the [Ray](https://www.ray.io/) backend for running multiple chunks across 
multiple processes. Best suited for accelerating functions that do not release GIL, such as regular 
Python functions. Ever wanted to test billions of hyperparameter combinations in a matter of minutes? 
This is now possible by scaling functions and entire applications up in the cloud using the Ray cluster. :eyes:

```pycon title="Benchmark sorting a matrix without and with multiprocessing"
>>> @vbt.chunked(
...     size=vbt.ArraySizer(arg_query='items', axis=1),
...     arg_take_spec=dict(
...         items=vbt.ArraySelector(axis=1)
...     ),
...     merge_func=np.column_stack
... )
... def bubble_sort(items):
...     items = items.copy()
...     for i in range(len(items)):
...         for j in range(len(items) - 1 - i):
...             if items[j] > items[j + 1]:
...                 items[j], items[j + 1] = items[j + 1], items[j]
...     return items
                
>>> items = np.random.uniform(size=(5000, 3))

>>> %time bubble_sort(items)
CPU times: user 11.9 s, sys: 111 ms, total: 12 s
Wall time: 12 s

>>> import ray; ray.init()
>>> %time bubble_sort(items, _engine='ray')
CPU times: user 51.8 ms, sys: 34.6 ms, total: 86.4 ms
Wall time: 4.32 s
```

### Jitting

- [x] Jitting means just-in-time compiling. In the vectorbt PRO universe though, jitting simply 
means accelerating. Although Numba remains the primary jitter, vectorbt PRO now enables implementation 
of custom jitter classes, such as that for vectorized NumPy and even [JAX](https://github.com/google/jax) 
with GPU support. Every jitted function is registered globally so you can switch between different 
implementations using a single command.

```pycon title="Run different implementations of cumulative sum"
>>> df = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... })
>>> df.vbt.cumsum()  # (1)!
    a   b
0   1   5
1   3   9
2   6  12
3  10  14
4  15  15

>>> df.vbt.cumsum(jitted=False)  # (2)!
    a   b
0   1   5
1   3   9
2   6  12
3  10  14
4  15  15

>>> @vbt.register_jitted(task_id_or_func=vbt.nb.nancumsum_nb)
... def nancumsum_np(arr):
...     return np.nancumsum(arr, axis=0)

>>> df.vbt.cumsum(jitted='np')  # (3)!
    a   b
0   1   5
1   3   9
2   6  12
3  10  14
4  15  15
```

1. Using the default function with Numba enabled
2. Using the default function with Numba disabled
3. Using our custom function

### Caching

- [x] Caching was reimplemented from the ground up and now it's being managed by a central registry. 
This allows for tracking useful statistics of all cacheable parts of vectorbt, such as to display the 
total cached size in MB. Full control and transparency.

```pycon title="Display caching statistics of a portfolio"
>>> price = vbt.YFData.fetch('BTC-USD').get('Close')
>>> pf = vbt.Portfolio.from_random_signals(price, n=5)
>>> _ = pf.stats()

>>> pf.get_ca_setup().get_status_overview(
...     filter_func=lambda setup: setup.caching_enabled,
...     include=['hits', 'misses', 'total_size']
... )
                                 hits  misses total_size
object                                                  
portfolio:0.drawdowns               0       1    70.9 kB
portfolio:0.exit_trades             0       1    70.5 kB
portfolio:0.filled_close            6       1    24.3 kB
portfolio:0.init_cash               3       1   32 Bytes
portfolio:0.init_position           0       1   32 Bytes
portfolio:0.init_position_value     0       1   32 Bytes
portfolio:0.init_value              5       1   32 Bytes
portfolio:0.input_value             1       1   32 Bytes
portfolio:0.orders                  9       1    69.7 kB
portfolio:0.total_profit            1       1   32 Bytes
portfolio:0.trades                  0       1    70.5 kB
```

### Hyperfast rolling metrics

- [x] Rolling metrics based on returns were optimized for best performance (up to 100x speedup).

```pycon title="Benchmark the rolling Sortino ratio"
>>> returns = pd.DataFrame(np.random.normal(0, 0.001, size=(1000, 100)))

>>> import quantstats as qs
>>> %timeit qs.stats.rolling_sortino(returns, rolling_period=10)  # (1)!
2.18 s ± 20 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> vbt.settings.wrapping['freq'] = '1d'
>>> %timeit returns.vbt.returns.rolling_sortino_ratio(window=10)  # (2)!
7.26 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

1. Using QuantStats
2. Using vectorbt PRO

## Flexibility

### Smart broadcasting

- [x] Broadcasting mechanism was completely refactored and now supports parameters. Build a product 
of multiple hyperparameter combinations with a single line of code :magic_wand:

```pycon title="Backtest the Golden Cross with different configurations of TSL and TP"
>>> price = vbt.YFData.fetch('BTC-USD').get('Close')
>>> fast_ma = vbt.MA.run(price, 50, short_name='fast_ma')
>>> slow_ma = vbt.MA.run(price, 200, short_name='slow_ma')
>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(
...     price, entries, exits, 
...     tsl_stop=pd.Index([np.nan, 0.1, 0.2]),
...     tp_stop=pd.Index([np.nan, 0.1, 0.2]),
... )
>>> pf.sharpe_ratio
tsl_stop  tp_stop
NaN       NaN        1.216473
          0.1       -0.001308
          0.2        0.208434
0.1       NaN        0.305441
          0.1        0.082011
          0.2        0.198531
0.2       NaN        0.930212
          0.1        0.043703
          0.2        0.381851
Name: sharpe_ratio, dtype: float64
```

### Smart array creation

- [x] Tired of manually constructing arrays and setting their data with Pandas? There is a functionality
to make your life much easier! Broadcaster now takes an index dictionary containing instructions on
where to set values in the array, and does the filling job for you. It knows exactly which axis has
to be modifed and doesn't create a full array if not necessary, with much care for RAM.

```pycon title="Compare accumulation daily and exit on Sunday, and accumulation weekly and exit on month end"
>>> price = vbt.YFData.fetch(['BTC-USD', 'ETH-USD']).get('Close')

>>> tile = pd.Index(["daily", "weekly"], name="strategy")  # (1)!
>>> pf = vbt.Portfolio.from_orders(
...     price,
...     size=vbt.index_dict({  # (2)!
...         vbt.ElemIdx(
...             vbt.RowPoints(every="D"), 
...             vbt.ColIdx("daily", level="strategy")): 100,  # (3)!
...         vbt.ElemIdx(
...             vbt.RowPoints(every="W-SUN"), 
...             vbt.ColIdx("daily", level="strategy")): -np.inf,  # (4)!
...         vbt.ElemIdx(
...             vbt.RowPoints(every="W-MON"), 
...             vbt.ColIdx("weekly", level="strategy")): 100,
...         vbt.ElemIdx(
...             vbt.RowPoints(every="M"), 
...             vbt.ColIdx("weekly", level="strategy")): -np.inf,
...     }),
...     size_type="value",
...     direction="longonly",
...     init_cash="auto",
...     broadcast_kwargs=dict(tile=tile)
... )
>>> pf.sharpe_ratio
strategy  symbol 
daily     BTC-USD    0.753833
          ETH-USD    0.832737
weekly    BTC-USD    0.859790
          ETH-USD    0.503182
Name: sharpe_ratio, dtype: float64
```

1. To represent two strategies, we need to tile the same data twice. For this,
create an index with strategy names and pass it as `tile` to the broadcaster for it to tile
the columns of each array (such as price) by two times.
2. Index dictionary contains index instructions as keys and data as values to be set.
Keys can be anything from row indices and labels, to custom indexer classes such as 
[RowPoints](/api/base/indexing/#vectorbtpro.base.indexing.RowPoints).
3. Find the indices of the rows that correspond to the beginning of each day and the index of the 
column "daily", and set each element under those indices to 100 (= accumulate)
4. Find the indices of the rows that correspond to Sunday. If any value under those indices has already 
been set with any previous instruction, it will be overridden.

### Index alignment

- [x] There is no more limitation of each Pandas array being required to have the same index.
Indexes of all arrays that should broadcast against each other are automatically aligned, as long as they
have the same data type.

```pycon title="Predict ETH price with BTC price using linear regression"
>>> btc_data = vbt.YFData.fetch('BTC-USD')
>>> btc_data.wrapper.shape
(2817, 7)

>>> eth_data = vbt.YFData.fetch('ETH-USD')  # (1)!
>>> eth_data.wrapper.shape
(1668, 7)

>>> ols = vbt.OLS.run(  # (2)!
...     btc_data.get("Close"),
...     eth_data.get("Close")
... )
>>> ols.pred
Date
2014-09-17 00:00:00+00:00            NaN
2014-09-18 00:00:00+00:00            NaN
2014-09-19 00:00:00+00:00            NaN
2014-09-20 00:00:00+00:00            NaN
2014-09-21 00:00:00+00:00            NaN
...                                  ...
2022-05-30 00:00:00+00:00    2109.769242
2022-05-31 00:00:00+00:00    2028.856767
2022-06-01 00:00:00+00:00    1911.555689
2022-06-02 00:00:00+00:00    1930.169725
2022-06-03 00:00:00+00:00    1882.573170
Freq: D, Name: Close, Length: 2817, dtype: float64
```

1. ETH-USD history is shorter than BTC-USD history
2. This now works! Just make sure that all arrays share the same timeframe and timezone.

### Meta methods

- [x] Many methods such as rolling apply are now available in two flavors: regular (instance methods) 
and meta (class methods). Regular methods are bound to a single array and do not have to take metadata 
anymore, while meta methods are not bound to any array and act as micro-pipelines with their own 
broadcasting and templating logic.

```pycon title="Compute the rolling z-score"
>>> @njit
... def zscore_nb(x):  # (1)!
...     return (x[-1] - np.mean(x)) / np.std(x)

>>> ts = pd.Series([1, 5, 2, 4, 3])

>>> ts.rolling(3).apply(zscore_nb, raw=True)  # (2)!
0         NaN
1         NaN
2   -0.392232
3    0.267261
4    0.000000
dtype: float64

>>> ts.vbt.rolling_apply(3, zscore_nb)  # (3)!
0         NaN
1         NaN
2   -0.392232
3    0.267261
4    0.000000
dtype: float64

>>> @njit
... def zscore_meta_nb(from_i, to_i, col, x):  # (4)!
...     x_window = x[from_i:to_i, col]
...     return (x_window[-1] - np.mean(x_window)) / np.std(x_window)

>>> pd.Series.vbt.rolling_apply(  # (5)!
...     3, 
...     zscore_meta_nb, 
...     ts.vbt.to_2d_array(), 
...     wrapper=ts.vbt.wrapper
... )
0         NaN
1         NaN
2   -0.392232
3    0.267261
4    0.000000
dtype: float64
```

1. Access to the window only
2. Using Pandas
3. Using the regular method, which accepts the same function as pandas
4. Access to the whole array
5. Using the meta method, which accepts metadata and variable arguments

```pycon title="Compute the rolling correlation coefficient of two arrays"
>>> @njit
... def corr_meta_nb(from_i, to_i, col, a, b):
...     a_window = a[from_i:to_i, col]
...     b_window = b[from_i:to_i, col]
...     return np.corrcoef(a_window, b_window)[1, 0]

>>> a = pd.Series([1, 6, 2, 5, 3, 4], name='c1')
>>> b = pd.DataFrame({
...     'c2': pd.Series([1, 2, 3, 4, 5, 6]),
...     'c3': pd.Series([6, 5, 4, 3, 2, 1])
... })

>>> vbt.pd_acc.rolling_apply(
...     3, 
...     corr_meta_nb, 
...     vbt.Rep('a'),
...     vbt.Rep('b'),
...     broadcast_named_args=dict(a=a, b=b)
... )
         c2        c3
0       NaN       NaN
1       NaN       NaN
2  0.188982 -0.188982
3 -0.240192  0.240192
4  0.327327 -0.327327
5 -0.500000  0.500000
```

### Flexible portfolio attributes

- [x] Portfolio attributes can now be partly or even entirely computed from user-defined arrays. 
This allows great control of post-simulation analysis, for example, to override some simulation data, 
to test hyperparameters without having to re-simulate the entire portfolio, or to avoid repeated 
reconstruction when caching is disabled.

```pycon title="Compute returns from custom data"
>>> price = pd.Series([1, 2, 3, 4, 5])
>>> size = pd.Series([1, 0, 0, 0, 0])
>>> cash_deposits = pd.Series([0, 0, 50, 0, 0])
>>> pf = vbt.Portfolio.from_orders(
...     price, 
...     size=size,
...     init_cash=100,
...     cash_deposits=cash_deposits
... )
>>> pf.get_returns()  # (1)!
0    0.000000
1    0.010000
2    0.009901
3    0.006579
4    0.006536
dtype: float64

>>> pf.get_returns(init_value=50)  # (2)!
0    1.000000
1    0.010000
2    0.009901
3    0.006579
4    0.006536
dtype: float64

>>> vbt.Portfolio.get_returns(  # (3)!
...     init_value=50,
...     cash_deposits=pf.cash_deposits,
...     value=pf.value,
...     wrapper=pf.wrapper
... )
0    1.000000
1    0.010000
2    0.009901
3    0.006579
4    0.006536
dtype: float64
```

1. Call the instance method to use the simulated data only
2. Call the instance method to use the simulated data but change the initial value
3. Call the class method to manually provide all the relevant data

## Data

### Local data

- [x] Added data classes that specialize in loading data from local files, such as CSV and HDF5. 
Also, any data can be effortlessly saved locally.

```pycon title="Generate random data, save separately, and load all at once"
>>> rand_data1 = vbt.RandomData.fetch('R1', start='2020-01-01', end='2020-01-05')
>>> rand_data2 = vbt.RandomData.fetch('R2', start='2020-01-02', end='2020-01-05')
>>> rand_data3 = vbt.RandomData.fetch('R3', start='2020-01-03', end='2020-01-05')

>>> rand_data1.to_hdf()
>>> rand_data2.to_hdf()
>>> rand_data3.to_hdf()

>>> hdf_data1 = vbt.HDFData.fetch('RandomData.h5/R1')  # (1)!
>>> hdf_data1.get()
2019-12-31 23:00:00+00:00     99.551987
2020-01-01 23:00:00+00:00     98.075728
2020-01-02 23:00:00+00:00     98.501028
2020-01-03 23:00:00+00:00     99.224922
2020-01-04 23:00:00+00:00    100.119515
Freq: D, dtype: float64

>>> hdf_data = vbt.HDFData.fetch('RandomData.h5')  # (2)!
```

1. vectorbt PRO supports parsing of directories, path expressions, and HDF keys
2. Automatically discovers and imports all keys in an HDF file

[=100% "Key 3/3"]{: .candystripe}

```pycon
>>> hdf_data.get()
symbol                             R1         R2          R3
2019-12-31 23:00:00+00:00   99.551987        NaN         NaN
2020-01-01 23:00:00+00:00   98.075728  98.667415         NaN
2020-01-02 23:00:00+00:00   98.501028  97.040207  100.433736
2020-01-03 23:00:00+00:00   99.224922  95.878958  100.070329
2020-01-04 23:00:00+00:00  100.119515  96.278695  101.840087
```

### Remote data

- [x] Supports more data classes for pulling remote data, such as from [Polygon.io](https://polygon.io/), 
[Alpha Vantage](https://www.alphavantage.co/), [Nasdaq Data Link](https://data.nasdaq.com/), and [TradingView](https://www.tradingview.com/).
Allows for gradually collecting and persisting any remote data.

```pycon
>>> data = vbt.NDLData.fetch('NSE/OIL')
>>> data.get()
                              Open     High      Low    Close    Close  \\
Date                                                                     
2009-09-30 00:00:00+00:00  1096.00  1156.70  1090.00  1135.00  1141.20   
2009-10-01 00:00:00+00:00  1102.00  1173.70  1102.00  1167.00  1166.35   
2009-10-05 00:00:00+00:00  1152.00  1165.90  1136.60  1143.00  1140.55   
...                            ...      ...      ...      ...      ...   
2019-01-02 00:00:00+00:00   175.80   176.20   171.00   172.35   172.40   
2019-01-03 00:00:00+00:00   172.80   175.70   171.50   172.00   172.00   
2019-01-04 00:00:00+00:00   172.05   174.95   172.05   174.55   174.55   

                           Total Trade Quantity  Turnover (Lacs)  
Date                                                              
2009-09-30 00:00:00+00:00            19748012.0        223877.07  
2009-10-01 00:00:00+00:00             3074254.0         35463.78  
2009-10-05 00:00:00+00:00              919832.0         10581.13  
...                                         ...              ...  
2019-01-02 00:00:00+00:00              722532.0          1251.85  
2019-01-03 00:00:00+00:00              698190.0          1212.34  
2019-01-04 00:00:00+00:00              431122.0           749.99  

[2299 rows x 7 columns]
```

## Indicators

### Plotting TA-Lib

- [x] Every TA-Lib indicator knows how to be plotted - fully automatically based on output flags.

```pycon
>>> data = vbt.YFData.fetch('BTC-USD', start='2020-01-01', end='2021-01-01')

>>> vbt.talib('MACD').run(data.get('Close')).plot()
```

![](/assets/images/features/talib.svg)

### Expressions

- [x] Indicators can be parsed from expressions. The indicator factory can derive all 
the required information such as inputs, parameters, outputs, and even NumPy, vectorbt, and 
TA-Lib functions and indicators thanks to annotations and a built-in matching mechanism.
Designing indicators has never been easier!

```pycon title="Create a MACD indicator"
>>> data = vbt.YFData.fetch('BTC-USD', start='2020-01-01', end='2021-01-01')

>>> expr = """
... MACD:
... fast_ema = @talib_ema(close, @p_fast_w)
... slow_ema = @talib_ema(close, @p_slow_w)
... macd = fast_ema - slow_ema
... signal = @talib_ema(macd, @p_signal_w)
... macd, signal
... """
>>> MACD = vbt.IF.from_expr(expr, fast_w=12, slow_w=26, signal_w=9)  # (1)!
>>> macd = MACD.run(data.get('Close'))
>>> fig = macd.macd.rename('MACD').vbt.plot()
>>> macd.signal.rename('Signal').vbt.plot(fig=fig)
```

1. No more manually passing `input_names`, `param_names`, and other information

![](/assets/images/features/macd.svg)

### WorldQuant Alphas

- [x] vectorbt PRO supports all [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) :eyes:

```pycon title="Run the first alpha"
>>> close = vbt.YFData.fetch(['BTC-USD', 'ETH-USD']).get('Close')

>>> vbt.wqa101(1).run(close).out
symbol                     BTC-USD  ETH-USD
Date                                       
2014-09-17 00:00:00+00:00     0.25     0.25
2014-09-18 00:00:00+00:00     0.25     0.25
2014-09-19 00:00:00+00:00     0.25     0.25
...                            ...      ...
2022-01-23 00:00:00+00:00     0.25     0.25
2022-01-24 00:00:00+00:00     0.50     0.00
2022-01-25 00:00:00+00:00     0.50     0.00

[2688 rows x 2 columns]
```

## Modeling

### Initial position

- [x] Similar to initial cash, initial position can now be specified.

```pycon title="Start with one Bitcoin"
>>> price = vbt.YFData.fetch('BTC-USD').get('Close')
>>> vbt.Portfolio.from_orders(price, init_cash=0, init_position=1).final_value
47585.25390625
```

### Cash deposits

- [x] Cash can now be deposited/withdrawn at any time. 

```pycon title="DCA $10 into Bitcoin each month"
>>> price = vbt.YFData.fetch('BTC-USD').get('Close')
>>> cash_deposits = pd.Series.vbt.empty_like(price, fill_value=0)
>>> month_start_mask = ~price.index.to_period('m').duplicated()
>>> cash_deposits[month_start_mask] = 10

>>> pf = vbt.Portfolio.from_orders(
...     price, 
...     init_cash=0, 
...     cash_deposits=cash_deposits
... )

>>> pf.input_value  # (1)!
510.0

>>> pf.final_value
2559.303734693575
```

1. Invested a total of $510

### Cash earnings

- [x] Cash earnings and dividends can now be added/removed at any time.

```pycon title="Backtest Apple without and with dividend reinvestment"
>>> data = vbt.YFData.fetch('AAPL', start='2010-01-01')

>>> pf_kept = vbt.Portfolio.from_holding(  # (1)!
...     data.get('Close'),
...     cash_dividends=data.get('Dividends')
... )

>>> pf_kept.cash.iloc[-1]  # (2)!
91.275667213787

>>> pf_kept.assets.iloc[-1]  # (3)!
15.497651762336607

>>> pf_reinvested = vbt.Portfolio.from_orders(  # (4)!
...     data.get('Close'),
...     cash_dividends=data.get('Dividends')
... )

>>> pf_reinvested.cash.iloc[-1]
0.0

>>> pf_reinvested.assets.iloc[-1]
18.30596343778988

>>> fig = pf_kept.value.rename('Value (kept)').vbt.plot()
>>> pf_reinvested.value.rename('Value (reinvested)').vbt.plot(fig=fig)
```

1. Keep dividends as cash
2. Final cash balance
3. Final number of shares in the portfolio
4. Reinvest dividends at the next bar

![](/assets/images/features/dividends.svg)

### Default order function

- [x] There is a new order function that behaves just like 
[Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders). 
No more boilerplate code for backtesting regular order data with custom callbacks!

```pycon title="Track exposure during simulation"
>>> @njit
... def post_order_func_nb(c, exposure):
...     position_value = c.position_now * c.val_price_now
...     exposure[c.i] = position_value / (position_value + c.cash_now)

>>> price = pd.Series([1, 2, 3, 2, 1]).astype(np.float_)
>>> size = pd.Series([np.nan, 0.5, np.nan, 0.5, np.nan])
>>> exposure = np.empty_like(price)
>>> pf = vbt.Portfolio.from_def_order_func(
...     price,
...     size=size,
...     size_type='targetpercent',
...     post_order_func_nb=post_order_func_nb,
...     post_order_args=(exposure,)
... )
>>> pf.wrapper.wrap(exposure)
0    0.000000
1    0.500000
2    0.600000
3    0.500000
4    0.333333
dtype: float64
```

### In-outputs

- [x] Portfolio can now take and return any user-defined arrays filled during the simulation, 
such as signals. In-output arrays can broadcast together with regular arrays. Additionally, 
vectorbt PRO will (semi-) automatically figure out how to correctly wrap and index the array, 
for example, whenever you select a column in the entire portfolio.

```pycon title="Track debt during simulation"
>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         col_debt = c.last_debt[col]
...         c.in_outputs.debt[c.i, col] = col_debt
...         if col_debt > c.in_outputs.max_debt[col]:
...             c.in_outputs.max_debt[col] = col_debt

>>> price = pd.DataFrame({
...     'a': pd.Series([10, 11, 12, 11, 10, 9, 8, 7]),
...     'b': pd.Series([7, 8, 9, 10, 11, 12, 11, 10])
... }).astype(np.float_)
>>> size = pd.Series([-1, 1, -1, 1, -1, 1, -1, 1])
>>> pf = vbt.Portfolio.from_def_order_func(
...     price, 
...     size=size,
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         debt=vbt.RepEval('np.empty_like(close)'),
...         max_debt=vbt.RepEval('np.full(close.shape[1], 0.)')
...     )  # (1)!
... )
>>> pf.get_in_output('debt')  # (2)!
      a     b
0  10.0   7.0
1   0.0   0.0
2  12.0   9.0
3   0.0   0.0
4  10.0  11.0
5   0.0   0.0
6   8.0  11.0
7   0.0   0.0

>>> pf.get_in_output('max_debt')  # (3)!
a    12.0
b    11.0
Name: max_debt, dtype: float64

>>> pf['b'].get_in_output('debt')  # (4)!
0     7.0
1     0.0
2     9.0
3     0.0
4    11.0
5     0.0
6    11.0
7     0.0
Name: b, dtype: float64

>>> pf['b'].get_in_output('max_debt')
11.0
```

1. Tell portfolio class to wait until all arrays are broadcast and create a new floating array of the final shape
2. Portfolio instance knows how to properly wrap a custom NumPy array into a pandas object
3. The same holds for reduced NumPy arrays
4. Portfolio instance also knows how to properly select a column/group in a custom NumPy array

### Portfolio optimization

- [x] There is a full-blown class for portfolio optimization that accepts a custom optimization function, 
and even integrates with [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/) 
and [Universal Portfolios](https://github.com/Marigold/universal-portfolios).

```pycon title="Optimize portfolio using Hierarchical Risk Parity on a monthly basis"
>>> close = vbt.BinanceData.fetch(
...     ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
...     start="2020-01-01",
...     end="2022-01-01",
...     missing_index="drop"
... ).get('Close')

>>> pf_opt = vbt.PortfolioOptimizer.from_pypfopt(
...     close.vbt.wrapper,
...     optimizer="hrp",
...     target="optimize",
...     prices=vbt.RepEval(
...         "close.iloc[index_slice]", 
...         context=dict(close=close)
...     ), 
...     every="MS"
... )
>>> pf_opt.plot()
```

[=100% "Allocation 15/15"]{: .candystripe}

![](/assets/images/features/hrp.svg)

### New order types

- [x] [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
has been extended with a variety of new order types, such as stop loss, trailing stop loss, trailing take profit, 
take profit, stop-limit, delta-limit, delayed, and time-in-force.

```pycon title="Explore how the limit delta affects the number of orders"
>>> pf = vbt.Portfolio.from_random_signals(
...     vbt.YFData.fetch('BTC-USD'), 
...     n=100, 
...     order_type="limit", 
...     limit_delta=pd.Index(np.arange(0.001, 0.1, 0.001)),  # (1)!
...     seed=42
... )
>>> pf.orders.count().vbt.plot(
...     xaxis_title="Limit delta", 
...     yaxis_title="Order count"
... )
```

1. Limit delta is the distance between the close (or any other price) and the target limit price
in percentage terms. The highest the delta, the less is the chance that it will be eventually hit.

![](/assets/images/features/limit_delta.svg)

## Analysis

### Shortcut properties

- [x] [In-output arrays](#in-outputs) can be used to override regular portfolio attributes. Portfolio 
will automatically pick the pre-computed array and perform all future calculations using this array, 
without wasting time on its reconstruction.

```pycon title="Modify returns from within simulation"
>>> @njit
... def post_segment_func_nb(c):
...     for col in range(c.from_col, c.to_col):
...         return_now = c.last_return[col]
...         return_now = 0.5 * return_now if return_now > 0 else return_now
...         c.in_outputs.returns[c.i, col] = return_now

>>> price = pd.Series([1, 2, 3, 2, 1]).astype(np.float_)
>>> size = pd.Series([np.nan, 0.5, np.nan, 0.5, np.nan])
>>> pf = vbt.Portfolio.from_def_order_func(
...     price,
...     size=size,
...     size_type='targetpercent',
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(
...         returns=vbt.RepEval("np.empty_like(close)")
...     )
... )

>>> pf.returns  # (1)!
0    0.000
1    0.000
2    0.125  << modified
3   -0.200
4   -0.250
dtype: float64

>>> pf.get_returns()  # (2)!
0    0.00
1    0.00
2    0.25
3   -0.20
4   -0.25
dtype: float64
```

2. Pre-computed returns are automatically taken from `in_outputs.returns`
3. Actual returns can still be reconstructed

### Benchmark

- [x] Benchmark can be easily set for the entire portfolio.

```pycon title="Compare Microsoft to S&P 500"
>>> data = vbt.YFData.fetch(['SPY', 'MSFT'], start='2010-01-01')

>>> pf = vbt.Portfolio.from_holding(
...     close=data.data['MSFT']['Close'],
...     bm_close=data.data['SPY']['Close']
... )
>>> pf.plot_cum_returns()
```

[=100% "Data 2/2"]{: .candystripe}

![](/assets/images/features/benchmark.svg)

### Multiple time frames

- [x] The [look-ahead bias](https://www.investopedia.com/terms/l/lookaheadbias.asp) is an ongoing 
threat when working with array data, especially on multiple time frames. In vectorbt PRO, there is 
an entire collection of functions for resampling and analyzing data in a safe way. Additionally,
there is a time frame parameter in each TA-Lib indicator!

```pycon title="Plot a heatmap of EMA on multiple time frames"
>>> h1_data = vbt.BinanceData.fetch(
...     'BTCUSDT', 
...     start="2020-01-01", 
...     end="2020-06-01", 
...     timeframe="1h")

>>> mtf_sma = vbt.talib("SMA").run(
...     h1_data.get("Close"), 
...     timeperiod=10, 
...     timeframe=["1h", "4h", "1d"], 
...     skipna=True, 
...     broadcast_kwargs=dict(
...         wrapper_kwargs=dict(freq="1h")
...     ))

>>> mtf_sma.real.iloc[:, ::-1].vbt.ts_heatmap()
```

[=100% "Period 8/8"]{: .candystripe}

![](/assets/images/features/mtf_heatmap.svg)

### Merging

- [x] Complex vectorbt objects of the same type can be easily merged along columns.
For instance, you can combine multiple totally-unrelated trading strategies into the same portfolio 
for analysis. Under the hood, the final object is still represented as a monolithic multi-dimensional
structure that can be processed even faster than merged objects separately!

```pycon title="Analyze two trading strategies separately and jointly"
>>> def strategy1(data):
...     fast_ma = vbt.MA.run(data.get("Close"), 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.get("Close"), 200, short_name='slow_ma')
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.Portfolio.from_signals(
...         data.get("Close"), 
...         entries, 
...         exits, 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> def strategy2(data):
...     bbands = vbt.BBANDS.run(data.get("Close"), window=14)
...     entries = bbands.close_crossed_below(bbands.lower)
...     exits = bbands.close_crossed_above(bbands.upper)
...     return vbt.Portfolio.from_signals(
...         data.get("Close"), 
...         entries, 
...         exits, 
...         init_cash=200
...     )

>>> data1 = vbt.BinanceData.fetch("BTCUSDT")
>>> pf1 = strategy1(data1)  # (1)!
>>> pf1.sharpe_ratio
0.9100317671866922

>>> data2 = vbt.BinanceData.fetch("ETHUSDT")
>>> pf2 = strategy2(data2)  # (2)!
>>> pf2.sharpe_ratio
-0.11596286232734827

>>> pf_sep = vbt.Portfolio.column_stack((pf1, pf2))  # (3)!
>>> pf_sep.sharpe_ratio
0    0.910032
1   -0.115963
Name: sharpe_ratio, dtype: float64

>>> pf_join = vbt.Portfolio.column_stack((pf1, pf2), group_by=True)  # (4)!
>>> pf_join.sharpe_ratio
0.42820898354646514
```

1. Analyze the first strategy in a separate portfolio
2. Analyze the second strategy in a separate portfolio
3. Analyze both strategies in the same portfolio separately
4. Analyze both strategies in the same portfolio jointly

### Appending

- [x] Complex vectorbt objects of the same type can be easily concatenated along rows.
For instance, you can append new data to an existing portfolio, or even concatenate in-sample 
portfolios with their out-of-sample counterparts :material-connection:

```pycon title="Analyze two date ranges separately and jointly"
>>> def strategy(data, start=None, end=None):
...     fast_ma = vbt.MA.run(data.get("Close"), 50, short_name="fast_ma")
...     slow_ma = vbt.MA.run(data.get("Close"), 200, short_name='slow_ma')
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     return vbt.Portfolio.from_signals(
...         data.get("Close")[start:end], 
...         entries[start:end], 
...         exits[start:end], 
...         size=100,
...         size_type="value",
...         init_cash="auto"
...     )

>>> data = vbt.BinanceData.fetch("BTCUSDT")

>>> pf_whole = strategy(data)  # (1)!
>>> pf_whole.sharpe_ratio
0.9100317671866922

>>> pf_sub1 = strategy(data, end="2019-12-31")  # (2)!
>>> pf_sub1.sharpe_ratio
0.7810397448678937

>>> pf_sub2 = strategy(data, start="2020-01-01")  # (3)!
>>> pf_sub2.sharpe_ratio
1.070339534746574

>>> pf_join = vbt.Portfolio.row_stack((pf_sub1, pf_sub2))  # (4)!
>>> pf_join.sharpe_ratio
0.9100317671866922
```

1. Analyze the entire range
2. Analyze the first date range
3. Analyze the second date range
4. Join both date ranges and analyze as a whole

### Slicing

- [x] Similarly to selecting columns, each vectorbt object is now capable of selecting rows, using the
exact same mechanism as in Pandas. This makes it supereasy to analyze and plot any subset of simulated data, 
without the need of re-simulation!

```pycon title="Analyze multiple date ranges of the same portfolio"
>>> data = vbt.YFData.fetch('BTC-USD')
>>> pf = vbt.Portfolio.from_holding(data, freq="d")

>>> pf.sharpe_ratio
1.116727709477293

>>> pf.loc[:"2020"].sharpe_ratio  # (1)!
1.2699801554196481

>>> pf.loc["2021": "2021"].sharpe_ratio  # (2)!
0.9825161170278687

>>> pf.loc["2022":].sharpe_ratio  # (3)!
-1.0423271337174647
```

1. Get the Sharpe during the year 2020 and before
2. Get the Sharpe during the year 2021
3. Get the Sharpe during the year 2022 and after

### More metrics

- [x] Portfolio, trades, drawdowns, and other vectorbt objects offer more analysis capabilities to choose from.

```pycon title="Analyze the MFE of a random portfolio without take profit"
>>> data = vbt.YFData.fetch('BTC-USD')

>>> pf = vbt.Portfolio.from_random_signals(data, n=50)
>>> pf.trades.plot_mfe_returns()
```

![](/assets/images/features/mfe_without_tp.svg)

```pycon title="Analyze the MFE of a random portfolio with take profit"
>>> pf = vbt.Portfolio.from_random_signals(data, n=50, tp_stop=0.1)
>>> pf.trades.plot_mfe_returns()
```

![](/assets/images/features/mfe_with_tp.svg)

## And many more...

- [x] This is just the tip of the iceberg: vectorbt PRO deploys a new project structure, modular 
settings, time and memory profiling tools, rich templating macros, an upgraded formatting engine, 
dynamic type checking, new efficient data structures, and more.
- [ ] Expect more killer features to be added on a weekly basis! :heart:{ .heart }


