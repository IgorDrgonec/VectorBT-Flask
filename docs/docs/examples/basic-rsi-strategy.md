---
title: Basic RSI strategy
---

# Basic RSI strategy

One of the main powers of vectorbt is the ability to create and backtest numerous strategy configurations
in the blink of an eye. In this introductory example, we will explore how profitable is the following 
RSI strategy commonly used by beginners:

> If the RSI is less than 30, it indicates a stock is reaching oversold conditions and may see 
> a trend reversal, or bounceback, towards a higher share price. Once the reversal is confirmed, 
> a buy trade is placed. Conversely, if the RSI is more than 70, it indicates that a stock is 
> reaching an overbought condition and may see a trend reversal, or pullback, in price. 
> After a confirmation of the reversal, a sell trade is placed.

As a bonus, we will gradually expand the analysis towards multiple 
[hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) combinations. 
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

![](/assets/images/examples_rsi_ohlcv.svg)

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
the Python's `help` function:

```pycon
>>> help(vbt.RSI.run)
Help on method run:

run(close, window=Default(value=14), ewm=Default(value=False), short_name='rsi', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
    Run `RSI` indicator.
    
    * Inputs: `close`
    * Parameters: `window`, `ewm`
    * Outputs: `rsi`
    
    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.
    
    Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
```

As we see above, we need to at least provide `close`, which can be any numeric time series. 
Also, by default, the rolling window is 14 days long and not exponential.
Since we want to make decisions using the opening price, we will pass `open_price` as `close`:

```pycon
>>> rsi = vbt.RSI.run(open_price)
>>> rsi
<vectorbtpro.indicators.custom.RSI at 0x7f9c20921ac8>
```

That's all! By executing the method [RSI.run](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.run), 
we calculated the RSI values and received an instance with various methods and properties for their 
analysis. To retrieve the resulting Pandas object, we need to query the `rsi` attribute (see "Outputs" 
in the output of `help`).

```pycon
>>> rsi.rsi
Open time
2017-08-17 00:00:00+00:00          NaN
2017-08-18 00:00:00+00:00          NaN
2017-08-19 00:00:00+00:00          NaN
...                                ...
2022-01-16 00:00:00+00:00    27.800150
2022-01-17 00:00:00+00:00    28.975756
2022-01-18 00:00:00+00:00    28.889430
Freq: D, Name: Open, Length: 1616, dtype: float64
```

!!! note
    If you compare RSI values of vectorbt and TA-Lib, you will notice that they differ.
    There are [different smoothing methods](https://www.macroption.com/atr-calculation/): 
    vectorbt uses SMA and EMA, while other libraries and TradingView use the original Wilder's method. 
    There is no right or wrong method. If you want to use the Wilder's method, use `vbt.talib('RSI')`.

Having the RSI array, we now want to generate an entry signal whenever any RSI value crosses below 30
and an exit signal whenever any RSI value crosses above 70:

```pycon
>>> entries = rsi.rsi.vbt.crossed_below(30)  # (1)!
>>> entries
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
...                            ...
2022-01-16 00:00:00+00:00    False
2022-01-17 00:00:00+00:00    False
2022-01-18 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1616, dtype: bool

>>> exits = rsi.rsi.vbt.crossed_above(70)  # (2)!
>>> exits
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
...                            ...
2022-01-16 00:00:00+00:00    False
2022-01-17 00:00:00+00:00    False
2022-01-18 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1616, dtype: bool
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
...     entries.vbt.signals.plot_as_entry_markers(rsi.rsi, fig=fig)  # (2)!
...     exits.vbt.signals.plot_as_exit_markers(rsi.rsi, fig=fig)  # (3)!
...     return fig

>>> plot_rsi(rsi, entries, exits)
```

1. Using [RSI.plot](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI.plot)
2. Using [SignalsSRAccessor.plot_as_entry_markers](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entry_markers)
3. Using [SignalsSRAccessor.plot_as_exit_markers](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exit_markers)

![](/assets/images/examples_rsi_rsi.svg)

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

![](/assets/images/examples_rsi_rsi2.svg)

We can immediately see the difference. But what other methods exist to analyze the distribution 
of signals? How to *quantify* such analysis? That's what vectorbt is all about. Let's
compute various statistics of `clean_entries` and `clean_exits` using
[SignalsAccessor](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsAccessor):

```pycon
>>> clean_entries.vbt.signals.total()  # (1)!
16

>>> clean_exits.vbt.signals.total()  # (2)!
15

>>> ranges = clean_entries.vbt.signals.between_ranges(other=clean_exits)  # (3)!
>>> ranges.duration.mean(wrap_kwargs=dict(to_timedelta=True))  # (4)!
Timedelta('40 days 01:36:00')
```

1. Get the total number of entry signals
2. Get the total number of exit signals
3. Get range records of type [Ranges](/api/generic/ranges/#vectorbtpro.generic.ranges.Ranges) 
between each entry and exit
4. Get the average duration between each entry and exit

We are ready for modeling! We will be using 
the class method [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
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

!!! hint
    If you look into the API of [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
    you will find many arguments to be set to None. `None` has a special meaning that instructs
    vectorbt to pull the default value from the global settings. You can discover all the default 
    values for the `Portfolio` class [here](/api/_settings/#vectorbtpro._settings.portfolio).

Let's print the statistics of our portfolio:

```pycon
>>> pf.stats()
Start                         2017-08-17 00:00:00+00:00
End                           2022-01-18 00:00:00+00:00
Period                               1616 days 00:00:00
Start Value                                  157.043476
End Value                                     110.98007
Total Return [%]                             -29.331627
Benchmark Return [%]                         877.704734
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                               58.04434
Max Drawdown Duration                1473 days 00:00:00
Total Trades                                         16
Total Closed Trades                                  15
Total Open Trades                                     1
Open Trade PnL                               -27.190641
Win Rate [%]                                  53.333333
Best Trade [%]                                35.794383
Worst Trade [%]                              -37.951988
Avg Winning Trade [%]                         12.256316
Avg Losing Trade [%]                         -16.703328
Avg Winning Trade Duration             19 days 03:00:00
Avg Losing Trade Duration              64 days 00:00:00
Profit Factor                                  0.838588
Expectancy                                    -1.258184
Sharpe Ratio                                  -0.014499
Calmar Ratio                                  -0.129933
Omega Ratio                                    0.996355
Sortino Ratio                                 -0.019584
dtype: object
```

!!! hint
    That's a lot of statistics, right? If you are looking for the way they are implemented,
    print `pf.metrics` and look for the `calc_func` argument of the metric of interest.
    If some function is a lambda, look at the source code to reveal its contents.

As you might have guessed, our strategy is great... at losing money (which is fascinating
given that Bitcoin increased 10x in value in the same timeframe). Even though the majority of the 
trades are profitable, we are losing much more on average than winning. Let's get some visuals 
for confirmation:

```pycon
>>> pf.plot(settings=dict(bm_returns=False))
```

![](/assets/images/examples_rsi_pf.svg)

!!! hint
    A benefit of an interactive plot like above is that you can use tools from the Plotly toolbar
    to draw a vertical line that connects orders, their P&L, and how they affect the cumulative returns.
    Try it out!

So, how do we improve from here?

## Multiple backtests

### Using for-loop

Even such a basic strategy as ours has many potential hyperparameters:

1. Lower threshold (`lower_th`)
2. Upper threshold (`upper_th`)
3. Window length (`window`)
4. Smoothing method (`ewm`)

To make our analysis as flexible as possible, we will write a function that lets us 
specify all of that information, and return a subset of statistics:

```pycon
>>> def test_rsi(window=14, ewm=False, lower_th=30, upper_th=70):
...     rsi = vbt.RSI.run(open_price, window=window, ewm=ewm)
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
Total Return [%]   -29.331627
Total Trades               16
Win Rate [%]        53.333333
Expectancy          -1.258184
dtype: object

>>> test_rsi(lower_th=20, upper_th=80)
Total Return [%]    59.691762
Total Trades                8
Win Rate [%]        71.428571
Expectancy           9.492297
dtype: object
```

!!! note
    We removed the signal cleaning step because it makes no difference when signals are
    passed to [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
    (which cleans the signals automatically anyway).

Bingo! The 80/20 configuration has done the trick. But how do we actually know whether this 
positive result indicates alpha and not because of a pure luck? Testing one hyperparameter 
combination from a huge space usually means making a wild guess.

Let's generate multiple hyperparameter combinations for thresholds, simulate them, and 
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
0           -7.493795             9     50.000000   -1.399830
1           -5.727722             9     62.500000   -1.074616
2           14.667515             9     75.000000    1.955225
3           48.225119             9     62.500000    6.774067
4           58.286605             9     62.500000    8.141884
..                ...           ...           ...         ...
116         30.188204            14     53.846154    4.688153
117         14.869720            13     50.000000    3.717156
118          9.517970            12     45.454545    3.698104
119         19.771377            12     54.545455    4.790308
120         22.317752            12     54.545455    5.088901

[121 rows x 4 columns]
```

But how do we know which row corresponds to which hyperparameter combination? 
We will build a [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
with two levels, `lower_th` and `upper_th`, and make it the index of `comb_stats_df`:

```pycon
>>> comb_stats_df.index = pd.MultiIndex.from_tuples(
...     th_combs, 
...     names=['lower_th', 'upper_th'])
>>> comb_stats_df
                   Total Return [%]  Total Trades  Win Rate [%]  Expectancy
lower_th upper_th                                                          
20       70               -7.493795             9     50.000000   -1.399830
         71               -5.727722             9     62.500000   -1.074616
         72               14.667515             9     75.000000    1.955225
         73               48.225119             9     62.500000    6.774067
         74               58.286605             9     62.500000    8.141884
...                             ...           ...           ...         ...
30       76               30.188204            14     53.846154    4.688153
         77               14.869720            13     50.000000    3.717156
         78                9.517970            12     45.454545    3.698104
         79               19.771377            12     54.545455    4.790308
         80               22.317752            12     54.545455    5.088901

[121 rows x 4 columns]
```

Much better! We can now analyze every piece of the retrieved information from different angles.
Since we have the same number of lower and upper thresholds, let's create a heatmap
with the X axis reflecting the lower thresholds, the Y axis reflecting the upper thresholds,
and the color bar reflecting the expectancy:

```pycon
>>> comb_stats_df['Expectancy'].vbt.heatmap()
```

![](/assets/images/examples_rsi_heatmap.svg)

We can observe entire regions of hyperparameter combinations that yield positive results.

### Using columns

As you might have read in [Fundamentals](/documentation/fundamentals), vectorbt loves processing
multi-dimensional data. In particular, it's built around the idea that you can represent
each asset, period, hyperparameter combination, and a backtest in general, as a column
in a two-dimensional array. 

Instead of computing everything in a loop (which isn't too bad but usually executes magnitudes 
slower than a vectorized solution) we can change our code to accept parameters as arrays. 
A function that takes such array will automatically convert multiple hyperparameters into 
multiple columns. A big benefit of this approach is that we don't have to collect our results, 
put them in a list, and convert into a DataFrame - it's all done by vectorbt!

First, define the hyperparameters that we would like to test:

```pycon
>>> windows = list(range(8, 21))
>>> ewms = [False, True]
>>> lower_ths = list(range(20, 31))
>>> upper_ths = list(range(70, 81))
```

Instead of applying `itertools.product`, we will instruct various parts of our pipeline to build 
a product instead, so we can observe how each part affects the column hierarchy.

The RSI part is easy: we can pass `param_product=True` to build a product of `windows` and `ewms` 
and run the calculation over each column in `open_price`:

```pycon
>>> rsi = vbt.RSI.run(
...     open_price, 
...     window=windows, 
...     ewm=ewms, 
...     param_product=True)
>>> rsi.rsi.columns
MultiIndex([( 8, False),
            ( 8,  True),
            ( 9, False),
            ( 9,  True),
            ...
            (19, False),
            (19,  True),
            (20, False),
            (20,  True)],
           names=['rsi_window', 'rsi_ewm'], length=26)
```

We see that [RSI](/api/indicators/custom/#vectorbtpro.indicators.custom.RSI) appended
two levels to the column hierarchy: `rsi_window` and `rsi_ewm`. Those are similar to the ones
we created manually for thresholds in [Using for-loop](#using-for-loop). There are now 26
columns in total, which is just `len(open_price.columns)` x `len(windows)` x `len(ewms)`.

The next part are crossovers. In contrast to indicators, they are regular functions that take
any array-like object, broadcast it to the `rsi` array, and search for crossovers.
The broadcasting step is done using [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast),
which is a very powerful function for bringing multiple arrays to a single shape
(read more in [Broadcasting](/documentation/fundamentals/#broadcasting)).

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
MultiIndex([(20,  8, False),
            (20,  8,  True),
            (20,  9, False),
            (20,  9,  True),
            ...
            (30, 19, False),
            (30, 19,  True),
            (30, 20, False),
            (30, 20,  True)],
           names=['lower_th', 'rsi_window', 'rsi_ewm'], length=3146)

>>> upper_th_index = pd.Index(upper_ths_prod, name='upper_th')
>>> exits = rsi.rsi_crossed_above(upper_th_index)
>>> exits.columns
MultiIndex([(70,  8, False),
            (70,  8,  True),
            (70,  9, False),
            (70,  9,  True),
            ...
            (80, 19, False),
            (80, 19,  True),
            (80, 20, False),
            (80, 20,  True)],
           names=['upper_th', 'rsi_window', 'rsi_ewm'], length=3146)
```

1. The first value in `lower_ths_prod` builds a combination with the first value in `upper_ths_prod`,
the second with the second, and so on - 121 combinations in total.
2. Convert thresholds to `pd.Index` to instruct [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast)
that we want to build a product with the columns in `rsi`

We have produced over 3146 columns - madness! But did you notice that `entries` and `exits` 
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
lower_th upper_th rsi_window rsi_ewm                                   
20       70       8          False          -19.280788            27   
                             True             1.618221            26   
                  9          False          -44.746787            22   
                             True            -1.752696            22   
                  10         False          -47.868781            16   
...                                                ...           ...   
30       80       18         True            29.819008             8   
                  19         False           -0.786254             5   
                             True            51.376108             8   
                  20         False          -37.263640             4   
                             True            33.669485             8   

                                      Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_ewm                            
20       70       8          False       53.846154   -0.512684  
                             True        64.000000    0.182982  
                  9          False       52.380952   -3.810420  
                             True        66.666667    0.043674  
                  10         False       60.000000   -5.769271  
...                                            ...         ...  
30       80       18         True        57.142857    6.865879  
                  19         False       75.000000    6.385719  
                             True        57.142857    9.508749  
                  20         False       33.333333  -16.159733  
                             True        57.142857    7.032204  

[3146 rows x 4 columns]
```

1. By default, [StatsBuilderMixin.stats](/api/generic/stats_builder/#vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats) 
takes the mean out of all columns and returns a Series. We, on the other hand, 
want to disable the aggregation function and stack all Series into one big DataFrame.

Congrats! We just backtested 3146 hyperparameter combinations in a quarter of a second :zap:

!!! important
    Even though we gained some unreal performance, we need to be careful to not occupy the entire RAM
    with our wide arrays. We can check the size of any [Pickleable](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable)
    instance using [Pickleable.getsize](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable.getsize).
    For example, to print the total size of our portfolio in a human-readable format:

    ```pycon
    >>> print(pf.getsize())
    45.1 MB
    ```

    We are still far away from the limit!

One option is to use Pandas itself to analyze the produced statistics. For example, 
calculate the mean expectancy of each `rsi_window`:

```pycon
>>> stats_df['Expectancy'].groupby('rsi_window').mean()
rsi_window
8     0.559154
9    -0.874399
10   -2.544668
11   -2.014821
12   -2.104681
13    1.824517
14    0.933219
15    3.287030
16    3.350205
17    6.763109
18    4.374821
19    3.515993
20    4.419658
Name: Expectancy, dtype: float64
```

The longer is the RSI window, the higher is the mean expectancy.

Display the top 5 hyperparameter combinations:

```pycon
>>> stats_df.sort_values(by='Expectancy', ascending=False).head()
                                      Total Return [%]  Total Trades  \\
lower_th upper_th rsi_window rsi_ewm                                   
24       74       20         True           136.770984             7   
23       74       20         True           136.770984             7   
24       77       17         False          115.904286             7   
         78       17         False          115.229435             7   
23       75       19         True           133.213236             7   

                                      Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_ewm                            
24       74       20         True        71.428571   19.538712  
23       74       20         True        71.428571   19.538712  
24       77       17         False       83.333333   19.288314  
         78       17         False       83.333333   19.175839  
23       75       19         True        71.428571   19.030462 
```

To analyze any particular combination using vectorbt, we can select it from the portfolio 
the same way as we selected a column in a regular Pandas DataFrame. Let's plot the equity 
of the most successful combination:

```pycon
>>> pf[(24, 74, 20, True)].plot_value()
```

![](/assets/images/examples_rsi_value.svg)

!!! hint
    Instead of selecting a column from a portfolio, which will create a new portfolio with only 
    that column, you can also check whether the method you want to call supports the argument `column`
    and pass your column using this argument. For instance, we could have also used 
    `pf.plot_value(column=(24, 74, 20, True))`.

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
MultiIndex([(20, 70,  8, False, 'BTCUSDT'),
            (20, 70,  8, False, 'ETHUSDT'),
            (20, 70,  8,  True, 'BTCUSDT'),
            (20, 70,  8,  True, 'ETHUSDT'),
            ...
            (30, 80, 20, False, 'BTCUSDT'),
            (30, 80, 20, False, 'ETHUSDT'),
            (30, 80, 20,  True, 'BTCUSDT'),
            (30, 80, 20,  True, 'ETHUSDT')],
           names=[
                'lower_th', 
                'upper_th', 
                'rsi_window', 
                'rsi_ewm', 
                'symbol'
           ], length=6292)
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

![](/assets/images/examples_rsi_histplot.svg)

ETH seems to react better to our strategy on average than BTC, maybe due to the market's 
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

[Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/notebooks/BasicRSI.ipynb){ .md-button target="blank_" }
