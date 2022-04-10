---
title: Fundamentals
icon: material/alphabetical-variant
---

# Fundamentals

vectorbt was implemented to address common performance shortcomings of backtesting libraries. 
It builds upon the idea that each instance of a trading strategy can be represented in a vectorized 
form, so multiple strategy instances can be packed into a single multi-dimensional array, processed 
in a highly efficient manner, and analyzed easily.

## Stack

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
[Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) (one-dimensional) and 
[DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) (two-dimensional). 
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
management cannot be simulated directly using vector calculations. We arrived at a point where we need 
to write a fast iterative code that processes data in an element-by-element fashion.

What if I told you that there exists a Python package that lets you run for-loops at machine code speed? 
And that it understands NumPy well and doesn't require adapting Python code much? It would solve many of 
our problems: our code could suddenly become incredibly fast while staying perfectly readable. 
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
array([nan, nan, 1., 2., 3., 4., 5., 6., 7., 8.])
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
    implements some hard-core stuff (:warning: adults only).

So where is the caveat? Sadly, Numba only understands NumPy, but not Pandas. This leaves
us without datetime index and other features so crucial for time series analysis. 
And that's where vectorbt comes into play: it replicates many Pandas functions using Numba 
and even adds some interesting features to them. This way, we not only make a subset of Pandas
faster, but also more powerful! 

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

## Accessors

Notice how `vbt` is attached directly to the Series object? This is called 
[an accessor](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors) -
a convenient way to extend Pandas objects without subclassing them. Using an accessor we can easily switch 
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

## Multidimensionality

Remember when we mentioned that vectorbt differs from traditional backtesters by taking and processing
trading data as multi-dimensional arrays? In particular, vectorbt treats each column as a separate 
backtesting instance rather than a feature. Consider a simple OHLC DataFrame:

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

Here, columns are separate features describing the same abstract object - price.
While it may appear intuitive to pass this DataFrame to vectorbt (as you may have done
with [scikit-learn](https://scikit-learn.org/stable/) and other ML tools, which expect 
DataFrames with features as columns), this approach has several key drawbacks in backtesting:

1. Features of different lengths/types are difficult to concatenate with NumPy
2. Optional features cannot be properly represented as arrays
3. Multiple backtests can be put into a 3-dimensional array (aka a cube), but it wouldn't be understood by Pandas
4. Non-changing features and constants would have had to be converted into arrays and
replicated across all backtests, leading to memory waste

vectorbt addresses this heterogeneity of features by processing them as separate arrays.
So instead of passing one big DataFrame, we need to provide each feature independently:

```pycon
>>> single_pf = vbt.Portfolio.from_holding(
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

Now, in case when we want to process multiple abstract objects, such as ticker symbols, 
we can simply pass DataFrames instead of Series:

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

>>> multi_pf = vbt.Portfolio.from_holding(
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

Here, each column (also often referred as "line" in vectorbt) in each feature DataFrame represents 
a separate backtesting instance and generates a separate equity curve. Thus, adding one more backtest 
is as simple as adding one more column to the features :sparkles:

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

The Pandas objects `multi_close` and `multi_open` can be Series and DataFrames of arbitrary shapes
and our micro-pipeline will still work as expected.

## Labels

In the example above, we created our multi-OHLC DataFrames with two columns - `p1` and `p2` -
so we can easily identify them later during the analysis phase. For this reason,
vectorbt ensures that those columns are preserved across the whole backtesting pipeline - 
from signal generation to performance modeling.

But what if individual columns corresponded to more complex configurations, such as those involving
multiple hyperparameter combinations? Storing complex objects as column labels wouldn't work after all.
Thanks to Pandas, there are [hierarchical columns](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html),
which are just like regular columns but stacked upon each other. Each level of such hierarchy
can help us to identify a specific input or parameter.

Take a simple crossover strategy as an example: it depends upon the lengths of the fast 
and slow windows. Each of these hyperparameters becomes an additional dimension for manipulating 
data and gets stored as a separate column level. Below is a more complex example of the column 
hierarchy of a MACD indicator:

```pycon
>>> macd = vbt.MACD.run(
...     multi_close,
...     fast_window=2,
...     slow_window=(3, 4),
...     signal_window=2,
...     macd_ewm=False,
...     signal_ewm=True
... )
>>> macd.signal
macd_fast_window          2      2      2      2  << fast window for MACD line
macd_slow_window          3      3      4      4  << slow window for MACD line
macd_signal_window        2      2      2      2  << window for signal line
macd_macd_ewm         False  False  False  False  << EWM flag for MACD line
macd_signal_ewm        True   True   True   True  << EWM flag for signal line
                         p1     p2     p1     p2  << price
2020-01-01              NaN    NaN    NaN    NaN
2020-01-02              NaN    NaN    NaN    NaN
2020-01-03              NaN    NaN    NaN    NaN
2020-01-04              0.5   -0.5    NaN    NaN
2020-01-05              0.5   -0.5    1.0   -1.0
```

The columns above capture two different backtesting configurations that can now be easily analyzed and 
compared using Pandas - a very powerful technique to analyze data. We may, for example, 
consider grouping our performance by `macd_fast_window` to see how the size of the fast window 
impacts the profitability of our strategy. Isn't this magic?

## Broadcasting

One of the most important concepts in vectorbt is broadcasting. Since vectorbt functions take 
time series as independent arrays, they need to know how to connect the elements of those arrays 
such that there is 1) complete information, 2) across all arrays, and 3) at each time step.

If all arrays are of the same size, vectorbt can easily perform any operation on an 
element-by-element basis. Whenever any of the arrays is of a smaller size though, vectorbt 
looks for a possibility to "stretch" it such that it can match the length of other arrays. 
This approach is heavily inspired by (and internally based upon) [NumPy's broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Why should we care about broadcasting? Because it allows us to pass array-like
objects of any shape to almost every function in vectorbt, be it constants or full-blown DataFrames,
and vectorbt will automatically figure out where the respective elements belong to.

```pycon
>>> part_arrays = dict(
...     close=pd.DataFrame({
...         'a': [1, 2, 3, 4], 
...         'b': [4, 3, 2, 1]
...     }),  # (1)!
...     size=pd.Series([1, -1, 1, -1]),  # (2)!
...     direction=['longonly', 'shortonly'],  # (3)!
...     fees=0.01  # (4)!
... )
>>> full_arrays = vbt.broadcast(part_arrays)

>>> full_arrays['close']
   a  b
0  1  4
1  2  3
2  3  2
3  4  1

>>> full_arrays['size']
   a  b
0  1  1
1 -1 -1
2  1  1
3 -1 -1

>>> full_arrays['direction']
          a          b
0  longonly  shortonly
1  longonly  shortonly
2  longonly  shortonly
3  longonly  shortonly

>>> full_arrays['fees']
      a     b
0  0.01  0.01
1  0.01  0.01
2  0.01  0.01
3  0.01  0.01
```

1. Per element
2. Per row
3. Per column
4. Per entire array

!!! hint
    As a rule of thumb:

    1. If any array is a Pandas object, always produces a Pandas object
    2. If any array is two-dimensional, always produces a two-dimensional array
    3. If all arrays are constants or one-dimensional, always produces a one-dimensional array
    4. If any array is a DataFrame and this array is a one-dimensional NumPy array, broadcasts along columns
    5. If this array is a Series, always broadcasts along rows
    5. Lists and other sequences are converted to NumPy arrays prior to broadcasting

In contrast to NumPy and Pandas, vectorbt knows how to broadcast labels: in case where columns 
or individual column levels in both objects are different, they are stacked upon each other. 
Consider checking whenever the fast moving average is higher than the slow moving average, 
using the following window combinations: (2, 3) and (3, 4).

```pycon
>>> fast_ma = vbt.MA.run(multi_close, window=[2, 3], short_name='fast')
>>> slow_ma = vbt.MA.run(multi_close, window=[3, 4], short_name='slow')

>>> fast_ma.ma
fast_window    2    2    3    3
              p1   p2   p1   p2
2020-01-01   NaN  NaN  NaN  NaN
2020-01-02   2.5  4.5  NaN  NaN
2020-01-03   3.5  3.5  3.0  4.0
2020-01-04   4.5  2.5  4.0  3.0
2020-01-05   5.5  1.5  5.0  2.0

>>> slow_ma.ma
slow_window    3    3    4    4
              p1   p2   p1   p2
2020-01-01   NaN  NaN  NaN  NaN
2020-01-02   NaN  NaN  NaN  NaN
2020-01-03   3.0  4.0  NaN  NaN
2020-01-04   4.0  3.0  3.5  3.5
2020-01-05   5.0  2.0  4.5  2.5

>>> fast_ma.ma > slow_ma.ma  # (1)!
ValueError: Can only compare identically-labeled DataFrame objects

>>> fast_ma.ma.values > slow_ma.ma.values  # (2)!
array([[False, False, False, False],
       [False, False, False, False],
       [ True, False, False, False],
       [ True, False,  True, False],
       [ True, False,  True, False]])

>>> fast_ma.ma.vbt > slow_ma.ma  # (3)!
fast_window      2      2      3      3
slow_window      3      3      4      4
                p1     p2     p1     p2
2020-01-01   False  False  False  False
2020-01-02   False  False  False  False
2020-01-03    True  False  False  False
2020-01-04    True  False   True  False
2020-01-05    True  False   True  False
```

1. Comparison with Pandas fails
2. Comparison with NumPy succeeds
3. Comparison with vectorbt succeeds

!!! hint
    Appending `.vbt` to a Pandas object on the left will broadcast both operands with 
    vectorbt and execute the operation with NumPy/Numba - ultimate combo :firecracker:

In contrast to Pandas, vectorbt broadcasts rows and columns by their absolute positions, not labels.
This broadcasting style is very similar to that of NumPy:

```pycon
>>> df1 = pd.DataFrame({'a': [0], 'b': [1]})
>>> df2 = pd.DataFrame({'b': [0], 'a': [1]})
>>> df1 + df2  # (1)!
   a  b
0  1  1

>>> df1.values + df2.values
array([[0, 2]])

>>> df1.vbt + df2  # (2)!
   a  b
   b  a
0  0  2
```

1. Pandas connects column `a` in `df1` and `df2`
2. vectorbt connects the first column in `df1` with the first column in `df2`, regardless of their labels

!!! important
    If you pass multiple arrays of data to vectorbt, ensure that their columns connect well positionally!

    In case your columns are not properly ordered, you will notice this by the result having
    multiple column levels with identical labels but different ordering.

Another feature of vectorbt is that it can broadcast objects with incompatible shapes but overlapping
multi-index levels - those having the same name or values. Continuing with the previous example, 
check whenever the fast moving average is higher than the price:

```pycon
>>> fast_ma.ma > multi_close  # (1)!
ValueError: Can only compare identically-labeled DataFrame objects

>>> fast_ma.ma.values > multi_close.values  # (2)!
ValueError: operands could not be broadcast together with shapes (5,4) (5,2) 

>>> fast_ma.ma.vbt > multi_close  # (3)!
fast_window      2      2      3      3
                p1     p2     p1     p2
2020-01-01   False  False  False  False
2020-01-02   False   True  False  False
2020-01-03   False   True  False   True
2020-01-04   False   True  False   True
2020-01-05   False   True  False   True
```

1. Comparison with Pandas fails
2. Comparison with NumPy fails
3. Comparison with vectorbt succeeds

And here comes more (bear with me): we can easily test multiple scalar-like hyperparameters
by passing them as a Pandas Index. Let's compare whether the price is within thresholds:

```pycon
>>> above_lower = multi_close.vbt > pd.Index([1, 2], name='lower')
>>> below_upper = multi_close.vbt < pd.Index([3, 4], name='upper')
>>> above_lower.vbt & below_upper
lower           1      1      2      2
upper           3      3      4      4
               p1     p2     p1     p2
2020-01-01   True  False  False  False
2020-01-02  False  False   True  False
2020-01-03  False  False  False   True
2020-01-04  False   True  False  False
2020-01-05  False  False  False  False
```

As you see, smart broadcasting is :gem: when it comes to merging information. 
See [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast) to learn more about
broadcasting principles and new exciting techniques to combine arrays.

## Flexible indexing

Broadcasting many big arrays consumes a lot of RAM and ultimately makes processing slower.
That's why vectorbt introduces a concept of "flexible indexing", which only returns the shape
to which arrays would have broadcast and, during processing, selects an element out of each 
array based on its current shape. A nice feature of this is that it has almost no additional 
memory footprint and can broadcast in any direction infinitely.

```pycon
>>> from vectorbtpro.base.indexing import flex_select_auto_nb

>>> a = np.array([1, 2, 3])
>>> b = np.array([[4], [5], [6]])
>>> c = np.array(10)

>>> flex_select_auto_nb(a, 2, 2)
3
>>> flex_select_auto_nb(b, 2, 2)
6
>>> flex_select_auto_nb(c, 2, 2)
10
```

This is the secret to how [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
manages to broadcast more than 50 arguments without losing any memory or performance :wink:
