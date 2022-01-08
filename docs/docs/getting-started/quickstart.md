---
title: Quickstart
---

# Quickstart

## Fundamentals

vectorbt was implemented to address common performance shortcomings of backtesting libraries. 
It builds upon the idea that each instance of a trading strategy can be represented in a vectorized 
form, so multiple strategy instances can be packed into a single multi-dimensional array, processed 
in a highly efficient manner, and analyzed easily.

### Stack

Thanks to the time-series nature of trading data, most of the aspects related to backtesting can be 
translated into arrays. In particular, vectorbt operates on [NumPy arrays](https://numpy.org/doc/stable/user/quickstart.html), 
which are ***very fast*** due to optimized, pre-compiled C code. NumPy arrays are supported by numerous 
scientific packages in the vibrant Python ecosystem, such as Pandas, NumPy, and Numba. There is a great 
chance that you already used some of those packages!

While NumPy excels at performance, it's not necessarily the most intuitive package for time series analysis. 
Consider the following moving average using NumPy:

```pycon
>>> import numpy as np

>>> def rolling_window(a, window):
...     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
...     strides = a.strides + (a.strides[-1],)
...     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

>>> np.mean(rolling_window(np.arange(10), 3), axis=1)
array([1., 2., 3., 4., 5., 6., 7., 8.])
```

While it might be ultrafast, it takes time for the user to understand what is going on and 
also some mastery to write such vectorized code without bugs. What about other 
rolling functions that are powering more complex indicators? And what about resampling, 
grouping, and other operations on dates and time?

Here comes [Pandas](https://pandas.pydata.org/docs/getting_started/overview.html) 
to the rescue! Pandas provides rich time series functionality, data alignment, NA-friendly statistics, 
groupby, merge and join methods, and lots of other conveniences. It has two primary data structures: 
[Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) (1-dimensional) and 
[DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) (2-dimensional). 
You can imagine them as NumPy arrays wrapped with valuable information, such as timestamps and column names.
Our moving average can be implemented in a one-liner:

```pycon
>>> import pandas as pd

>>> index = pd.date_range('2020-01-01', '2020-01-10')
>>> sr = pd.Series(range(len(index)), index=index)
>>> sr.rolling(3).mean()
2020-01-01    NaN
2020-01-02    NaN
2020-01-03    1.0
2020-01-04    2.0
2020-01-05    3.0
2020-01-06    4.0
2020-01-07    5.0
2020-01-08    6.0
2020-01-09    7.0
2020-01-10    8.0
Freq: D, dtype: float64
```

vectorbt relies heavily upon Pandas, but not in the way you think. Pandas has one big drawback for 
our use case: it's slow for bigger datasets and custom-defined functions. Many functions such as 
the rolling mean are implemented using [Cython](https://cython.org/) under the hood and are 
sufficiently fast. But once you try to implement a more complex function, such as some rolling 
ranking metric based on multiple time series, things are becoming complicated and slow. 
In addition, what about functions that cannot be vectorized? A portfolio strategy involving money 
management cannot be simulated directly using vector calculations.

What if I told you that there exists a Python package that lets you run for-loops at machine code speed?
And that it understands NumPy well and doesn't require adapting Python code much? It would solve
many of our problems: our code could suddenly become incredible fast while staying perfectly readable.
This package is [Numba](https://numba.pydata.org/numba-doc/latest/user/5minguide.html). 
Numba translates a subset of Python and NumPy code into fast machine code.

```pycon
>>> from numba import njit

>>> @njit
... def moving_average_nb(a, window_len):
...     b = np.empty_like(a, dtype=np.float_)
...     for i in range(len(a)):
...         window_start = max(0, i + 1 - window_len)
...         window_end = i + 1
...         if window_end - window_start < window_len:
...             b[i] = np.nan
...         else:
...             b[i] = np.mean(a[window_start:window_end])
...     return b

>>> moving_average_nb(np.arange(10), 3)
array([nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
```

We can now clearly understand what is going on: we iterate over our time series one timestamp 
at a time, check whether there is enough data in the window, and if there is, we take the mean of it. 
Not only Numba is great for writing a human-readable and less error-prone code, it's also as fast as 
[C](https://en.wikipedia.org/wiki/C_(programming_language))!

```pycon
>>> big_a = np.arange(1000000)
>>> %imeit moving_average_nb.py_func(big_a, 10)  # (1)!
6.54 s ± 142 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit np.mean(rolling_window(big_a, 10), axis=1)  # (2)!
24.7 ms ± 173 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit pd.Series(big_a).rolling(10).mean()  # (3)!
10.2 ms ± 309 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit moving_average_nb(big_a, 10)  # (4)!
5.12 ms ± 7.21 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

1. Python
2. NumPy
3. Pandas
4. Numba!

!!! hint
    If you're interested in how vectorbt uses Numba, just look at any directory or file
    with the name `nb`. [This module](https://github.com/polakowo/vectorbt.pro/blob/main/vectorbtpro/generic/nb.py) 
    implements all the basic functions, while [this module](https://github.com/polakowo/vectorbt.pro/blob/main/vectorbtpro/portfolio/nb/from_order_func.py)
    implements some hard-core stuff (warning: adults only).

So where is the caveat? Sadly, Numba only understands NumPy, but not Pandas. This leaves
us without datetime index and other features so crucial for time series analysis. 
And that's where vectorbt comes into play: it replicates many Pandas functions using Numba 
and even adds some interesting features to them. This way, we not only make a subset of Pandas
faster but also more powerful! 

This is done as follows:

1. Extract the NumPy array from the Pandas object
2. Run a Numba-compiled function on this array
3. Wrap the result back using Pandas

```pycon
>>> arr = sr.values
>>> result = moving_average_nb(arr, 3)
>>> new_sr = pd.Series(result, index=sr.index, name=sr.name)
>>> new_sr
2020-01-01    NaN
2020-01-02    NaN
2020-01-03    1.0
2020-01-04    2.0
2020-01-05    3.0
2020-01-06    4.0
2020-01-07    5.0
2020-01-08    6.0
2020-01-09    7.0
2020-01-10    8.0
Freq: D, dtype: float64
```

Or using vectorbt:

```pycon
>>> sr.vbt.rolling_mean(3)
2020-01-01    NaN
2020-01-02    NaN
2020-01-03    1.0
2020-01-04    2.0
2020-01-05    3.0
2020-01-06    4.0
2020-01-07    5.0
2020-01-08    6.0
2020-01-09    7.0
2020-01-10    8.0
Freq: D, dtype: float64
```

### Accessors

Notice how `vbt` is attached directly to the Pandas Series? This is called 
[an accessor](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors) -
a convenient way to extend Pandas objects without subclassing them. This way, we can easily switch 
between native Pandas and vectorbt functionality. Moreover, each vectorbt method is flexible towards 
inputs and can work on both Series and DataFrames.

```pycon
>>> df = pd.DataFrame({'a': range(10), 'b': range(9, -1, -1)})
>>> df.vbt.rolling_mean(3)
     a    b
0  NaN  NaN
1  NaN  NaN
2  1.0  8.0
3  2.0  7.0
4  3.0  6.0
5  4.0  5.0
6  5.0  4.0
7  6.0  3.0
8  7.0  2.0
9  8.0  1.0
```

You can learn more about vectorbt's accessors [here](/api/accessors/). For instance, `rolling_mean`
is part of the accessor [GenericAccessor](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor),
which can be accessed directly using `vbt`. Another popular accessor [ReturnsAccessor](/api/returns/accessors/#vectorbtpro.returns.accessors.ReturnsAccessor)
for processing returns is a subclass of `GenericAccessor` and can be accessed using `vbt.returns`.

```pycon
>>> ret = pd.Series([0.1, 0.2, -0.1])
>>> ret.vbt.returns.total()
0.18800000000000017
```

!!! important
    Each accessor expects the data to be in the ready-to-use format. This means that the accessor 
    for working with returns expects the data to be returns, not the price!

### Multidimensionality

Remember when we mentioned that vectorbt differs from traditional backtesters by taking and processing
trading data as multi-dimensional arrays? In particular, it views each column as a separate backtesting 
instance rather than a feature. Consider a simple OHLC DataFrame:

```pycon
>>> p1 = pd.DataFrame({
...     'open': [1, 2, 3, 4, 5],
...     'high': [0.5, 1.5, 2.5, 3.5, 4.5],
...     'low': [2.5, 3.5, 4.5, 5.5, 6.5],
...     'close': [2, 3, 4, 5, 6]
... }, index=pd.date_range('2020-01-01', '2020-01-05'))
>>> p1
            open  high  low  close
2020-01-01     1   0.5  2.5      2
2020-01-02     2   1.5  3.5      3
2020-01-03     3   2.5  4.5      4
2020-01-04     4   3.5  5.5      5
2020-01-05     5   4.5  6.5      6
```

Here, columns are separate features describing the same object - price.
While it may appear intuitive to pass this DataFrame to vectorbt (as you may have done
with [scikit-learn](https://scikit-learn.org/stable/) and other ML tools, which expect 
DataFrames with features as columns), this approach has several key drawbacks in backtesting:

1. Features of different lengths/types are difficult to concatenate with NumPy
2. Optional features cannot be properly represented as arrays
3. Multiple backtests can be put into a 3-dimensional array (aka a cube) but it wouldn't be understood by Pandas
4. Non-changing features and constants would have had to be converted into arrays and
replicated across all backtests, leading to memory waste

vectorbt addresses this heterogeneity of features by processing them as separate arrays.
So instead of passing one big DataFrame, we need to provide each feature independently:

```pycon
>>> single_pf = vbt.Portfolio.from_orders(
...     open=p1['open'], 
...     high=p1['high'], 
...     low=p1['low'], 
...     close=p1['close']
... )
>>> single_pf.value
2020-01-01    100.0
2020-01-02    150.0
2020-01-03    200.0
2020-01-04    250.0
2020-01-05    300.0
Freq: D, dtype: float64
```

Now, in the case where we want to process multiple data objects, we can simply pass DataFrames instead of Series:

```pycon
>>> p2 = pd.DataFrame({
...     'open': [6, 5, 4, 3, 2],
...     'high': [6.5, 5.5, 4.5, 3.5, 2.5],
...     'low': [4.5, 3.5, 2.5, 1.5, 0.5],
...     'close': [5, 4, 3, 2, 1]
... }, index=pd.date_range('2020-01-01', '2020-01-05'))
>>> p2
            open  high  low  close
2020-01-01     6   6.5  4.5      5
2020-01-02     5   5.5  3.5      4
2020-01-03     4   4.5  2.5      3
2020-01-04     3   3.5  1.5      2
2020-01-05     2   2.5  0.5      1

>>> multi_open = pd.DataFrame({
...     'p1': p1['open'],
...     'p2': p2['open']
... })
>>> multi_high = pd.DataFrame({
...     'p1': p1['high'],
...     'p2': p2['high']
... })
>>> multi_low = pd.DataFrame({
...     'p1': p1['low'],
...     'p2': p2['low']
... })
>>> multi_close = pd.DataFrame({
...     'p1': p1['close'],
...     'p2': p2['close']
... })

>>> multi_pf = vbt.Portfolio.from_orders(
...     open=multi_open,
...     high=multi_high,
...     low=multi_low,
...     close=multi_close
... )
>>> multi_pf.value
               p1     p2
2020-01-01  100.0  100.0
2020-01-02  150.0   80.0
2020-01-03  200.0   60.0
2020-01-04  250.0   40.0
2020-01-05  300.0   20.0
```

Here, each column in each feature DataFrame represents a separate backtesting instance and 
generates a separate equity curve. Thus, adding one more backtest is as simple as adding 
one more column to all features :sparkles:

Keeping features separated has another big advantage: we can combine them easily.
And not only this: we combine all backtesting instances at once using vectorization.
Consider the following example where we place an entry signal whenever the previous candle 
was green and an exit signal whenever the previous candle was red (which is pretty dumb but anyway):

```pycon
>>> candle_green = multi_close > multi_open
>>> prev_candle_green = candle_green.vbt.signals.fshift(1)
>>> prev_candle_green
               p1     p2
2020-01-01  False  False
2020-01-02   True  False
2020-01-03   True  False
2020-01-04   True  False
2020-01-05   True  False

>>> candle_red = multi_close < multi_open
>>> prev_candle_red = candle_red.vbt.signals.fshift(1)
>>> prev_candle_red
               p1     p2
2020-01-01  False  False
2020-01-02  False   True
2020-01-03  False   True
2020-01-04  False   True
2020-01-05  False   True
```

### To be continued... :eyes: