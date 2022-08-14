---
title: Basic RSI strategy
description: Learn about backtesting a basic RSI strategy
---

# Basic RSI strategy

One of the main powers of vectorbt (PRO) is the ability to create and backtest numerous strategy 
configurations in the blink of an eye. In this introductory example, we will explore how profitable 
is the following RSI strategy commonly used by beginners:

> If the RSI is less than 30, it indicates a stock is reaching oversold conditions and may see 
> a trend reversal, or bounceback, towards a higher share price. Once the reversal is confirmed, 
> a buy trade is placed. Conversely, if the RSI is more than 70, it indicates that a stock is 
> reaching an overbought condition and may see a trend reversal, or pullback, in price. 
> After a confirmation of the reversal, a sell trade is placed.

As a bonus, we will gradually expand the analysis towards multiple parameter combinations. 
Sounds fun? Let's start. 

## Single backtest

First, we will take care of data. Using a one-liner, we will download all available daily data for 
the pair BTC/USDT from Binance:

```pycon
>>> import vectorbtpro as vbt
>>> import numpy as np
>>> import pandas as pd

>>> data = vbt.BinanceData.fetch('BTCUSDT')
>>> data
<vectorbtpro.data.custom.BinanceData at 0x7f9c40c59550>
```

[=100% "100%"]{: .candystripe}

The returned object is of type [BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData),
which extends [Data](/api/data/base/#vectorbtpro.data.base.Data) to communicate with the Binance API. 
The class [Data](/api/data/base/#vectorbtpro.data.base.Data) is a vectorbt's in-house container 
for retrieving, storing, and managing data. Upon receiving a DataFrame, it post-processes and 
stores the DataFrame inside the dictionary [Data.data](/api/data/base/#vectorbtpro.data.base.Data.data) 
keyed by pair (also referred to as a "symbol" in vectorbt). We can get our DataFrame either from this 
dictionary, or by using the convenient method [Data.get](/api/data/base/#vectorbtpro.data.base.Data.get), 
which also allows for specifying one or more columns instead of returning the entire DataFrame at once.

Let's pull the DataFrame and use the accessor 
[OHLCVDFAccessor](/api/ohlcv/accessors/#vectorbtpro.ohlcv.accessors.OHLCVDFAccessor) to plot it:

```pycon
>>> data.data['BTCUSDT'].vbt.ohlcv.plot()
```

![](/assets/images/tutorials/rsi/ohlcv.svg)

Another way to describe the data is by using the Pandas' 
[info](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) method. 
The tabular format is especially useful for counting null values (which our data apparently doesn't have - good!)

```pycon
>>> data.data['BTCUSDT'].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1616 entries, 2017-08-17 00:00:00+00:00 to 2022-01-18 00:00:00+00:00
Freq: D
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype              
---  ------              --------------  -----              
 0   Open                1616 non-null   float64            
 1   High                1616 non-null   float64            
 2   Low                 1616 non-null   float64            
 3   Close               1616 non-null   float64            
 4   Volume              1616 non-null   float64            
 5   Close time          1616 non-null   datetime64[ns, UTC]
 6   Quote volume        1616 non-null   float64            
 7   Number of trades    1616 non-null   int64              
 8   Taker base volume   1616 non-null   float64            
 9   Taker quote volume  1616 non-null   float64            
dtypes: datetime64[ns, UTC
```

In our example, we will generate signals based on the opening price and execute them based 
on the closing price. We can also place orders a soon as the signal is generated, or at any 
later time, but we will illustrate how to separate generation of signals from their execution.

```pycon
>>> open_price = data.get('Open')
>>> close_price = data.get('Close')
```

It's time to run the indicator!

vectorbt supports 4 (!) different implementations of RSI: one implemented using Numba, 
and the other three ported from three different technical analysis libraries. Each indicator
has been wrapped with the almighty 
[IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory) :mechanical_arm:

```pycon
>>> vbt.RSI  # (1)!
vectorbtpro.indicators.custom.RSI

>>> vbt.talib('RSI')  # (2)!
vectorbtpro.indicators.factory.talib.RSI

>>> vbt.ta('RSIIndicator')  # (3)!
vectorbtpro.indicators.factory.ta.RSIIndicator

>>> vbt.pandas_ta('RSI')  # (4)!
vectorbtpro.indicators.factory.pandas_ta.RSI
```

1. To list all vectorbt indicators, check out the module [custom](/api/indicators/custom/)
2. To list all supported TA-Lib indicators, call [IndicatorFactory.get_talib_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_talib_indicators)
3. To list all supported TA indicators, call [IndicatorFactory.get_ta_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_ta_indicators)
4. To list all supported Pandas TA indicators, call [IndicatorFactory.get_pandas_ta_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_pandas_ta_indicators)

Here's a rule of thumb on which implementation to choose:

1. Use TA-Lib indicators for fastest execution
2. Use vectorbt indicators for fast execution and plotting
3. Use indicators from other libraries in case they provide more options

To run any indicator, use the method `run`. To see what arguments the method accepts, pass it to 
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func):

```pycon
>>> print(vbt.format_func(vbt.RSI.run))
RSI.run(
    close,
    window=Default(value=14),
    wtype=Default(value='wilder'),
    short_name='rsi',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `RSI` indicator.
    
    * Inputs: `close`
    * Parameters: `window`, `wtype`
    * Outputs: `rsi`
    
    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.
    
    Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
```

As we can see above, we need to at least provide `close`, which can be any numeric time series. 
Also, by default, the rolling window is 14 bars long and uses the Wilder's smoothed moving average.
Since we want to make decisions using the opening price, we will pass `open_price` as `close`:

```pycon
>>> rsi = vbt.RSI.run(open_price)
>>> rsi
<vectorbtpro.indicators.custom.RSI at 0x7f9c20921ac8>
```

That's all! By executing the method [RSI.run](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.run), 
we calculated the RSI values and have received an instance with various methods and properties for their 
analysis. To retrieve the resulting Pandas object, we need to query the `rsi` attribute (see "Outputs" 
in the output of `format_func`).

```pycon
>>> rsi.rsi
Open time
2017-08-17 00:00:00+00:00          NaN
2017-08-18 00:00:00+00:00          NaN
2017-08-19 00:00:00+00:00          NaN
2017-08-20 00:00:00+00:00          NaN
2017-08-21 00:00:00+00:00          NaN
...                                ...
2022-07-30 00:00:00+00:00    60.541637
2022-07-31 00:00:00+00:00    59.503179
2022-08-01 00:00:00+00:00    56.750576
2022-08-02 00:00:00+00:00    56.512434
2022-08-03 00:00:00+00:00    54.177385
Freq: D, Name: Open, Length: 1813, dtype: float64
```

Having the RSI array, we now want to generate an entry signal whenever any RSI value crosses below 30
and an exit signal whenever any RSI value crosses above 70:

```pycon
>>> entries = rsi.rsi.vbt.crossed_below(30)  # (1)!
>>> entries
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool

>>> exits = rsi.rsi.vbt.crossed_above(70)  # (2)!
>>> exits
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool
```

1. Using [GenericAccessor.crossed_below](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_below)
2. Using [GenericAccessor.crossed_above](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_above)

The same can be done using the methods [RSI.rsi_crossed_below](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.rsi_crossed_below) 
and [RSI.rsi_crossed_above](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.rsi_crossed_above) 
that were auto-generated for the output `rsi` by [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory):

```pycon
>>> entries = rsi.rsi_crossed_below(30)
>>> exits = rsi.rsi_crossed_above(70)
```

!!! hint
    If you are curious what else has been generated, print `dir(rsi)` or look into the 
    [API](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI) generated for the class.

Before we proceed with the portfolio modeling, let's plot the RSI and signals to ensure that we 
did everything right:

```pycon
>>> def plot_rsi(rsi, entries, exits):
...     fig = rsi.plot()  # (1)!
...     entries.vbt.signals.plot_as_entries(rsi.rsi, fig=fig)  # (2)!
...     exits.vbt.signals.plot_as_exits(rsi.rsi, fig=fig)  # (3)!
...     return fig

>>> plot_rsi(rsi, entries, exits)
```

1. Using [RSI.plot](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.plot)
2. Using [SignalsSRAccessor.plot_as_entries](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entries)
3. Using [SignalsSRAccessor.plot_as_exits](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exits)

![](/assets/images/tutorials/rsi/rsi.svg)

The graph looks legit. But notice how there are multiple entries between two exits and vice versa?
How does vectorbt handle it? When using [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
vectorbt will automatically filter out all entry signals if the position has already been entered,
and exit signals if the position has already been exited. But to make our analysis cleaner, 
let's keep each first signal:

```pycon
>>> clean_entries, clean_exits = entries.vbt.signals.clean(exits)  # (1)!

>>> plot_rsi(rsi, clean_entries, clean_exits)
```

1. Using [SignalsAccessor.clean](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsAccessor.clean)

![](/assets/images/tutorials/rsi/rsi2.svg)

We can immediately see the difference. But what other methods exist to analyze the distribution 
of signals? How to *quantify* such analysis? That's what vectorbt is all about. Let's
compute various statistics of `clean_entries` and `clean_exits` using
[SignalsAccessor](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsAccessor):

```pycon
>>> clean_entries.vbt.signals.total()  # (1)!
8

>>> clean_exits.vbt.signals.total()  # (2)!
7

>>> ranges = clean_entries.vbt.signals.between_ranges(other=clean_exits)  # (3)!
>>> ranges.duration.mean(wrap_kwargs=dict(to_timedelta=True))  # (4)!
Timedelta('86 days 10:17:08.571428572')
```

1. Get the total number of entry signals
2. Get the total number of exit signals
3. Get range records of type [Ranges](/api/generic/ranges/#vectorbtpro.generic.ranges.Ranges) 
between each entry and exit
4. Get the average duration between each entry and exit

We are ready for modeling! We will be using the class method [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
which will receive the signal arrays, process each signal one by one, and generate orders. 
It will then create an instance of [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) 
that can be used to assess the performance of the strategy.

Our experiment is simple: buy $100 of Bitcoin upon an entry signal and close the position upon 
an exit signal. Start with an infinite capital to not limit our buying power at any time.

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=clean_entries, 
...     exits=clean_exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c40eea438>
```

!!! info
    Running the method above for the first time may take some time as it must be compiled first.
    Compilation will take place each time a new combination of data types is discovered.
    But don't worry: Numba caches most compiled functions and re-uses them in each new runtime.

!!! hint
    If you look into the API of [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
    you will find many arguments to be set to None. The value `None` has a special meaning that instructs
    vectorbt to pull the default value from the global settings. You can discover all the default 
    values for the `Portfolio` class [here](/api/_settings/#vectorbtpro._settings.portfolio).

Let's print the statistics of our portfolio:

```pycon
>>> pf.stats()
Start                         2017-08-17 00:00:00+00:00
End                           2022-08-03 00:00:00+00:00
Period                               1813 days 00:00:00
Start Value                                       100.0
Min Value                                     97.185676
Max Value                                    203.182943
End Value                                    171.335425
Total Return [%]                              71.335425
Benchmark Return [%]                         446.481746
Total Time Exposure [%]                       38.113624
Max Gross Exposure [%]                            100.0
Max Drawdown [%]                              46.385941
Max Drawdown Duration                1613 days 00:00:00
Total Orders                                         15
Total Fees Paid                                     0.0
Total Trades                                          8
Win Rate [%]                                  71.428571
Best Trade [%]                                54.519055
Worst Trade [%]                              -32.078597
Avg Winning Trade [%]                         26.905709
Avg Losing Trade [%]                         -19.345383
Avg Winning Trade Duration             87 days 09:36:00
Avg Losing Trade Duration              84 days 00:00:00
Profit Factor                                  3.477019
Expectancy                                    13.691111
Sharpe Ratio                                   0.505486
Calmar Ratio                                   0.246836
Omega Ratio                                    1.132505
Sortino Ratio                                  0.796701
dtype: object
```

!!! hint
    That are lots of statistics, right? If you're looking for the way they are implemented,
    print `pf.metrics` and look for the `calc_func` argument of the metric of interest.
    If some function is a lambda, look into the source code to reveal its contents.

Our strategy is not too bad: the portfolio has gained over 71% in profit over the last years, but holding 
Bitcoin is still better - staggering 450%. Despite the Bitcoin's high volatility, the minimum recorded
portfolio value sits at $97 from $100 initially invested. The total time exposure of 38% means that 
we were in the market 38% of the time. The maximum gross exposure of 100% means that we invested 100% of 
our available cash balance, each single trade. The maximum drawdown (MDD) of 46% is the maximum distance
our portfolio value fell after recording a new high (stop loss to the rescue?). 

The total number of orders matches the total number of (cleaned) signals, but why is the total number 
of trades suddenly 8 instead of 15? By default, a trade in the vectorbt's universe is a sell order; 
as soon as an exit order has been filled (by reducing or closing the current position), the profit 
and loss (PnL) based on the weighted average entry and exit price is calculated. The win rate of 70% 
means that 70% of the trades (sell orders) generated a profit, with the best trade bringing 54% in profit
and the worst one bringing 32% in loss. Since the average winning trade generating more profit
than the average losing trade generating loss, we can see various metrics being positive, such as 
the profit factor and the expectancy.

```pycon
>>> pf.plot(settings=dict(bm_returns=False))
```

![](/assets/images/tutorials/rsi/pf.svg)

!!! hint
    A benefit of an interactive plot like above is that you can use tools from the Plotly toolbar
    to draw a vertical line that connects orders, their P&L, and how they affect the cumulative returns.
    Try it out!

So, how do we improve from here?

## Multiple backtests

### Using for-loop

Even such a basic strategy as ours has many potential parameters:

1. Lower threshold (`lower_th`)
2. Upper threshold (`upper_th`)
3. Window length (`window`)
4. Smoothing method (`ewm`)

To make our analysis as flexible as possible, we will write a function that lets us 
specify all of that information, and return a subset of statistics:

```pycon
>>> def test_rsi(window=14, wtype="wilder", lower_th=30, upper_th=70):
...     rsi = vbt.RSI.run(open_price, window=window, wtype=wtype)
...     entries = rsi.rsi_crossed_below(lower_th)
...     exits = rsi.rsi_crossed_above(upper_th)
...     pf = vbt.Portfolio.from_signals(
...         close=close_price, 
...         entries=entries, 
...         exits=exits,
...         size=100,
...         size_type='value',
...         init_cash='auto')
...     return pf.stats([
...         'total_return', 
...         'total_trades', 
...         'win_rate', 
...         'expectancy'
...     ])

>>> test_rsi()
Total Return [%]    71.335425
Total Trades                8
Win Rate [%]        71.428571
Expectancy          13.691111
dtype: object

>>> test_rsi(lower_th=20, upper_th=80)
Total Return [%]    6.652287
Total Trades               2
Win Rate [%]            50.0
Expectancy          3.737274
dtype: object
```

!!! note
    We removed the signal cleaning step because it makes no difference when signals are
    passed to [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
    (which cleans the signals automatically anyway).

By raising the upper threshold to 80% and lowering the lower threshold to 20%, the number
of trades has decreased to just 2 because it becomes more difficult to cross the thresholds.
We can also observe how the total return fell to roughly 7% - not a good sign. But how do we actually 
know whether this negative result indicates that our strategy is trash and not because of a pure luck? 
Testing one parameter combination from a huge space usually means making a wild guess.

Let's generate multiple parameter combinations for thresholds, simulate them, and 
concatenate their statistics for further analysis:

```pycon
>>> from itertools import product

>>> lower_ths = range(20, 31)  # (1)!
>>> upper_ths = range(70, 81)  # (2)!
>>> th_combs = list(product(lower_ths, upper_ths))  # (3)!
>>> len(th_combs)
121

>>> comb_stats = [
...     test_rsi(lower_th=lower_th, upper_th=upper_th)
...     for lower_th, upper_th in th_combs
... ]  # (4)!
```

1. 20, 21, ..., 30
2. 70, 71, ..., 80
3. Generate all possible combinations between `lower_ths` and `upper_ths`,
also called as a [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product)
4. Iterate over each combination and compute its statistics using a
[list comprehension](https://realpython.com/list-comprehension-python/). This creates
a list of Series.

We just simulated 121 different combinations of the upper and lower threshold
and stored their statistics inside a list. In order to analyze this list,
we need to convert it to a DataFrame first, with metrics arranged as columns:

```pycon
>>> comb_stats_df = pd.DataFrame(comb_stats)
>>> comb_stats_df
     Total Return [%]  Total Trades  Win Rate [%]  Expectancy
0           24.369550             3     66.666667   10.606342
1           37.380341             3     66.666667   16.203667
2           34.560194             3     66.666667   14.981187
3           31.090080             3     66.666667   13.833710
4           31.090080             3     66.666667   13.833710
..                ...           ...           ...         ...
116         51.074571             6     80.000000   18.978193
117         62.853840             6     80.000000   21.334047
118         40.685579             5     75.000000   21.125494
119         -5.990835             4     66.666667   13.119897
120        -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

But how do we know which row corresponds to which parameter combination? 
We will build a [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
with two levels, `lower_th` and `upper_th`, and make it the index of `comb_stats_df`:

```pycon
>>> comb_stats_df.index = pd.MultiIndex.from_tuples(
...     th_combs, 
...     names=['lower_th', 'upper_th'])
>>> comb_stats_df
                   Total Return [%]  Total Trades  Win Rate [%]  Expectancy
lower_th upper_th                                                          
20       70               24.369550             3     66.666667   10.606342
         71               37.380341             3     66.666667   16.203667
         72               34.560194             3     66.666667   14.981187
         73               31.090080             3     66.666667   13.833710
         74               31.090080             3     66.666667   13.833710
...                             ...           ...           ...         ...
30       76               51.074571             6     80.000000   18.978193
         77               62.853840             6     80.000000   21.334047
         78               40.685579             5     75.000000   21.125494
         79               -5.990835             4     66.666667   13.119897
         80              -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

Much better! We can now analyze every piece of the retrieved information from different angles.
Since we have the same number of lower and upper thresholds, let's create a heatmap
with the X axis reflecting the lower thresholds, the Y axis reflecting the upper thresholds,
and the color bar reflecting the expectancy:

```pycon
>>> comb_stats_df['Expectancy'].vbt.heatmap()
```

![](/assets/images/tutorials/rsi/heatmap.svg)

We can explore entire regions of parameter combinations that yield positive or negative results.

### Using columns

As you might have read in the documentation, vectorbt loves processing multidimensional data. 
In particular, it's built around the idea that you can represent each asset, period, parameter 
combination, and a backtest in general, as a column in a two-dimensional array.

Instead of computing everything in a loop (which isn't too bad but usually executes magnitudes 
slower than a vectorized solution) we can change our code to accept parameters as arrays. 
A function that takes such array will automatically convert multiple parameters into 
multiple columns. A big benefit of this approach is that we don't have to collect our results, 
put them in a list, and convert into a DataFrame - it's all done by vectorbt!

First, define the parameters that we would like to test:

```pycon
>>> windows = list(range(8, 21))
>>> wtypes = ["simple", "exp", "wilder"]
>>> lower_ths = list(range(20, 31))
>>> upper_ths = list(range(70, 81))
```

Instead of applying `itertools.product`, we will instruct various parts of our pipeline to build 
a product instead, so we can observe how each part affects the column hierarchy.

The RSI part is easy: we can pass `param_product=True` to build a product of `windows` and `wtypes` 
and run the calculation over each column in `open_price`:

```pycon
>>> rsi = vbt.RSI.run(
...     open_price, 
...     window=windows, 
...     wtype=wtypes, 
...     param_product=True)
>>> rsi.rsi.columns
MultiIndex([( 8, 'simple'),
            ( 8,    'exp'),
            ( 8, 'wilder'),
            ...
            (20, 'simple'),
            (20,    'exp'),
            (20, 'wilder')],
           names=['rsi_window', 'rsi_wtype'])
```

We see that [RSI](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI) appended
two levels to the column hierarchy: `rsi_window` and `rsi_wtype`. Those are similar to the ones
we created manually for thresholds in [Using for-loop](#using-for-loop). There are now 39
columns in total, which is just `len(open_price.columns)` x `len(windows)` x `len(wtypes)`.

The next part are crossovers. In contrast to indicators, they are regular functions that take
any array-like object, broadcast it to the `rsi` array, and search for crossovers.
The broadcasting step is done using [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast),
which is a very powerful function for bringing multiple arrays to a single shape
(learn more about broadcasting in the documentation).

In our case, we want to build a product of `lower_ths`, `upper_th_index`, and all columns in `rsi`.
Since both `rsi_crossed_below` and `rsi_crossed_above` are two different functions,
we need to build a product of the threshold values manually and then instruct each 
crossover function to combine them with every column in `rsi`:

```pycon
>>> lower_ths_prod, upper_ths_prod = zip(*product(lower_ths, upper_ths))
>>> len(lower_ths_prod)  # (1)!
121
>>> len(upper_ths_prod)
121

>>> lower_th_index = pd.Index(lower_ths_prod, name='lower_th')  # (2)!
>>> entries = rsi.rsi_crossed_below(lower_th_index)
>>> entries.columns
MultiIndex([(20,  8, 'simple'),
            (20,  8,    'exp'),
            (20,  8, 'wilder'),
            ...
            (30, 20, 'simple'),
            (30, 20,    'exp'),
            (30, 20, 'wilder')],
           names=['lower_th', 'rsi_window', 'rsi_wtype'], length=4719)

>>> upper_th_index = pd.Index(upper_ths_prod, name='upper_th')
>>> exits = rsi.rsi_crossed_above(upper_th_index)
>>> exits.columns
MultiIndex([(70,  8, 'simple'),
            (70,  8,    'exp'),
            (70,  8, 'wilder'),
            ...
            (80, 20, 'simple'),
            (80, 20,    'exp'),
            (80, 20, 'wilder')],
           names=['upper_th', 'rsi_window', 'rsi_wtype'], length=4719)
```

1. The first value in `lower_ths_prod` builds a combination with the first value in `upper_ths_prod`,
the second with the second, and so on - 121 combinations in total.
2. Convert thresholds to `pd.Index` to instruct [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast)
that we want to build a product with the columns in `rsi`

We have produced over 4719 columns - madness! But did you notice that `entries` and `exits` 
have different columns now? The first one has `lower_th` as one of the column levels, the second one
has `upper_th`. How are we supposed to pass differently labeled arrays (including `close_price` with 
one column) to [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)?

No worries, vectorbt knows exactly how to merge this information. Let's see:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=entries, 
...     exits=exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c415ed5c0>

>>> stats_df = pf.stats([
...     'total_return', 
...     'total_trades', 
...     'win_rate', 
...     'expectancy'
... ], agg_func=None)  # (1)!
>>> stats_df
                                        Total Return [%]  Total Trades  \\
lower_th upper_th rsi_window rsi_wtype                                   
20       70       8          simple           -25.285842            31   
                             exp               -7.939736            29   
                             wilder            61.979801            11   
...                                                  ...           ...   
                  20         simple           -59.159157             4   
                             exp               -3.331163             8   
                             wilder            31.479482             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
20       70       8          simple        51.612903   -1.224523  
                             exp           58.620690   -0.307862  
                             wilder        72.727273    5.634527  
...                                              ...         ...  
                  20         simple        33.333333  -16.159733  
                             exp           57.142857    7.032204  
                             wilder        50.000000   38.861607  

[4719 rows x 4 columns]
```

1. By default, [StatsBuilderMixin.stats](/api/generic/stats_builder/#vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats) 
takes the mean out of all columns and returns a Series. We, on the other hand, 
want to disable the aggregation function and stack all Series into one big DataFrame.

Congrats! We just backtested 4719 parameter combinations in less than a second :zap:

!!! important
    Even though we gained some unreal performance, we need to be careful to not occupy the entire RAM
    with our wide arrays. We can check the size of any [Pickleable](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable)
    instance using [Pickleable.getsize](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable.getsize).
    For example, to print the total size of our portfolio in a human-readable format:

    ```pycon
    >>> print(pf.getsize())
    9.4 MB
    ```

    Even though the portfolio holds about 10 MB of compressed data, it must generate many arrays,
    such as the portfolio value, that have the same shape as the number of timestamps x parameter combinations:

    ```pycon
    >>> np.product(pf.wrapper.shape) * 8 / 1024 / 1024
    65.27364349365234
    ```

    We can see that each floating array occupies 65 MB of memory. By creating a dozen of such arrays 
    (which is often the worst case), the memory consumption may jump to 1 GB very quickly.

One option is to use Pandas itself to analyze the produced statistics. For example, 
calculate the mean expectancy of each `rsi_window`:

```pycon
>>> stats_df['Expectancy'].groupby('rsi_window').mean()
rsi_window
8      0.154425
9      0.064130
10    -0.915478
11    -0.523294
12     0.742266
13     3.898482
14     4.414367
15     6.916872
16     8.915225
17    12.204188
18    12.897135
19    14.508950
20    16.429515
Name: Expectancy, dtype: float64
```

The longer is the RSI window, the higher is the mean expectancy.

Display the top 5 parameter combinations:

```pycon
>>> stats_df.sort_values(by='Expectancy', ascending=False).head()
                                        Total Return [%]  Total Trades  \\
lower_th upper_th rsi_window rsi_wtype                                   
22       80       20         wilder           187.478208             2   
21       80       20         wilder           187.478208             2   
26       80       20         wilder           152.087039             3   
23       80       20         wilder           187.478208             2   
25       80       20         wilder           201.297495             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
22       80       20         wilder            100.0   93.739104  
21       80       20         wilder            100.0   93.739104  
26       80       20         wilder            100.0   93.739104  
23       80       20         wilder            100.0   93.739104  
25       80       20         wilder            100.0   93.739104  
```

To analyze any particular combination using vectorbt, we can select it from the portfolio 
the same way as we selected a column in a regular Pandas DataFrame. Let's plot the equity 
of the most successful combination:

```pycon
>>> pf[(22, 80, 20, "wilder")].plot_value()
```

![](/assets/images/tutorials/rsi/value.svg)

!!! hint
    Instead of selecting a column from a portfolio, which will create a new portfolio with only 
    that column, you can also check whether the method you want to call supports the argument `column`
    and pass your column using this argument. For instance, we could have also used 
    `pf.plot_value(column=(22, 80, 20, "wilder"))`.

Even though, in theory, the best found setting doubles our money, it's still inferior to 
simply holding Bitcoin - our basic RSI strategy cannot beat the market :anger:

But even if it did, there is much more to just searching for right parameters:
we need at least to (cross-) validate the strategy. We can also observe how the strategy
behaves on other assets. Curious how to do it? Just expand `open_price` and `close_price`
to contain multiple assets, and each example would work out-of-the-box!

```pycon
>>> data = vbt.BinanceData.fetch(['BTCUSDT', 'ETHUSDT'])
```

[=100% "100%"]{: .candystripe}

Your homework is to run the examples on this data.

The final columns should become as follows:

```pycon
MultiIndex([(20, 70,  8, 'simple', 'BTCUSDT'),
            (20, 70,  8, 'simple', 'ETHUSDT'),
            (20, 70,  8,    'exp', 'BTCUSDT'),
            ...
            (30, 80, 20,    'exp', 'ETHUSDT'),
            (30, 80, 20, 'wilder', 'BTCUSDT'),
            (30, 80, 20, 'wilder', 'ETHUSDT')],
           names=['lower_th', 'upper_th', 'rsi_window', 'rsi_wtype', 'symbol'], length=9438)
```

We see that the column hierarchy now contains another level - `symbol` - denoting the asset. 
Let's visualize the distribution of the expectancy across both assets:

```pycon
>>> eth_mask = stats_df.index.get_level_values('symbol') == 'ETHUSDT'
>>> btc_mask = stats_df.index.get_level_values('symbol') == 'BTCUSDT'
>>> pd.DataFrame({
...     'ETHUSDT': stats_df[eth_mask]['Expectancy'].values,
...     'BTCUSDT': stats_df[btc_mask]['Expectancy'].values
... }).vbt.histplot(xaxis=dict(title="Expectancy"))  # (1)!
```

1. Using [GenericAccessor.histplot](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.histplot)

![](/assets/images/tutorials/rsi/histplot.svg)

ETH seems to react more aggressively to our strategy on average than BTC, maybe due to the market's 
higher volatility, a different structure, or just pure randomness.

And here's one of the main takeaways of such analysis: using strategies with simple and 
explainable mechanics, we can try to explain the mechanics of the market itself. Not only can we 
use this to improve ourselves and design better indicators, but use this information as an input 
to ML models, which are better at connecting dots than humans. Possibilities are endless!

## Summary

vectorbt is a powerful vehicle that enables us to discover uncharted territories faster 
and analyze them in more detail. Instead of using overused and outdated charts and indicators 
from books and YouTube videos, we can build our own tools that go hand in hand with the market. 
We can backtest thousands of strategy configurations to learn how the market reacts to each one 
of them - in a matter of milliseconds. All it takes is creativity :bulb:

[:material-lock: Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/locked-notebooks.md){ .md-button target="blank_" }
