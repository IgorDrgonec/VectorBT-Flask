---
title: Parsers
icon: material/code-tags
---

# Parsers

[IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory) 
deploys a collection of parsers to simplify creation of indicators, ranging from third-party
indicator parsers to a powerful expression parser.

!!! info
    Each parser method is a class method with the prefix `from_`, meaning we don't have to 
    construct and pass any information to the indicator factory using `vbt.IF(...)` - 
    the method already does it for us!

## TA-Lib

[IndicatorFactory.from_talib](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_talib) 
can parse [TA-Lib](https://github.com/mrjbq7/ta-lib) indicators. Whenever we pass the name of an indicator,
the method gets the TA-Lib abstract function, and then looks in the `info` dictionary to derive the input, 
parameter, and output names. After constructing a factory instance, it builds an apply function that can 
run the indicator function on two-dimensional inputs (as opposed to one-dimensional inputs that are
currently supported by TA-Lib).

To get the list of all supported indicators:

```pycon
>>> import vectorbtpro as vbt

>>> vbt.IF.get_talib_indicators()  # (1)!
{'ACOS',
 'AD',
 'ADD',
 ...
 'WCLPRICE',
 'WILLR',
 'WMA'}
```

1. Using [IndicatorFactory.get_talib_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_talib_indicators) 

To get an indicator:

```pycon
>>> vbt.IF.from_talib('RSI')  # (1)!
vectorbtpro.indicators.factory.talib.RSI
```

1. Using [IndicatorFactory.from_talib](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_talib) 

Or, using a shortcut:

```pycon
>>> vbt.talib('RSI')
vectorbtpro.indicators.factory.talib.RSI
```

### Skipping NaN

TA-Lib indicators are usually very unhappy when they encounter missing data. For instance, 
a single NaN in a time series can propagate this NaN to all data points that follow:

```pycon
>>> import numpy as np

>>> price = vbt.RandomData.fetch(
...     start='2020-01-01', 
...     end='2020-06-01', 
...     freq='1H',
...     seed=42
... ).get()
>>> price_na = price.copy()
>>> price_na.iloc[2] = np.nan  # (1)!

>>> SMA = vbt.talib("SMA")
>>> sma = SMA.run(price_na, timeperiod=10)
>>> sma.real
2019-12-31 22:00:00+00:00   NaN
2019-12-31 23:00:00+00:00   NaN
2020-01-01 00:00:00+00:00   NaN
2020-01-01 01:00:00+00:00   NaN
2020-01-01 02:00:00+00:00   NaN
...                         ...
2020-05-31 18:00:00+00:00   NaN
2020-05-31 19:00:00+00:00   NaN
2020-05-31 20:00:00+00:00   NaN
2020-05-31 21:00:00+00:00   NaN
2020-05-31 22:00:00+00:00   NaN
Freq: H, Name: 10, Length: 3649, dtype: float64
```

1. Make a single value missing

To address this, we can tell the indicator factory to run the indicator on non-NA values only
and then place the output values at their original positions:

```pycon
>>> sma = SMA.run(price_na, timeperiod=10, skipna=True)
>>> sma.real
2019-12-31 22:00:00+00:00           NaN
2019-12-31 23:00:00+00:00           NaN
2020-01-01 00:00:00+00:00           NaN
2020-01-01 01:00:00+00:00           NaN
2020-01-01 02:00:00+00:00           NaN
...                                 ...
2020-05-31 18:00:00+00:00    213.169260
2020-05-31 19:00:00+00:00    212.477181
2020-05-31 20:00:00+00:00    211.911416
2020-05-31 21:00:00+00:00    211.310849
2020-05-31 22:00:00+00:00    210.899923
Freq: H, Name: 10, Length: 3649, dtype: float64
```

!!! hint
    Another option would be to forward fill NaN values before running an indicator, but this
    would skew the results, thus make use of this option whenever this is really appropriate.

### Resampling

Another feature implemented by the indicator factory is the support for parametrized time frames!

This works the following way: 

1. Using the wrapper, input arrays are downsampled to the target time frame
2. Indicator is run on downsampled input arrays
3. Output arrays are upsampled back to the original time frame

This way, multiple time frames can be packed into a single two-dimensional array:

```pycon
>>> sma = SMA.run(
...     price_na, 
...     timeperiod=10, 
...     skipna=True, 
...     timeframe=["1h", "4h", "1d"]
... )
>>> sma.real
sma_timeperiod                                             10
sma_timeframe                      1h          4h          1d
2019-12-31 22:00:00+00:00         NaN         NaN         NaN
2019-12-31 23:00:00+00:00         NaN         NaN         NaN
2020-01-01 00:00:00+00:00         NaN         NaN         NaN
2020-01-01 01:00:00+00:00         NaN         NaN         NaN
2020-01-01 02:00:00+00:00         NaN         NaN         NaN
...                               ...         ...         ...
2020-05-31 18:00:00+00:00  213.169260  215.561805  206.104351
2020-05-31 19:00:00+00:00  212.477181  214.422456  206.104351
2020-05-31 20:00:00+00:00  211.911416  214.422456  206.104351
2020-05-31 21:00:00+00:00  211.310849  214.422456  206.104351
2020-05-31 22:00:00+00:00  210.899923  214.422456  206.104351

[3649 rows x 3 columns]
```

!!! note
    When some timestamps are missing, vectorbt may have difficulties parsing the frequency
    of the source index. To provide the frequency explicitly, pass 
    `broadcast_kwargs=dict(wrapper_kwargs=dict(freq="1h"))`, for example.
    Without the source frequency, vectorbt will upsample the downsampled arrays 
    between each two timestamps in the source index instead of relying on its frequency,
    which may be undesired.

### Plotting

Additionally, we can plot each indicator. This is achieved fully programmatically by parsing the indicator's 
output flags. Let's take `STOCH` as an example:

```pycon
>>> STOCH = vbt.talib('STOCH')
>>> STOCH.output_flags
OrderedDict([('slowk', ['Dashed Line']), ('slowd', ['Dashed Line'])])

>>> ohlc = price.resample('1d').ohlc()
>>> stoch = STOCH.run(ohlc['high'], ohlc['low'], ohlc['close'])
>>> stoch.plot()
```

![](/assets/images/indicators_stoch.svg)

To see what arguments the `plot` function takes, use 
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func):

```pycon
>>> print(vbt.format_func(STOCH.plot))
plot(
    self,
    column=None,
    limits=None,
    add_shape_kwargs=None,
    add_trace_kwargs=None,
    fig=None,
    slowk_trace_kwargs=None,
    slowd_trace_kwargs=None,
    **layout_kwargs
):
    Plot the outputs of the indicator based on their flags.
            
    Args:
        column (str): Name of the column to plot.
        limits (tuple of float): Tuple of the lower and upper limit.
        add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
        add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
        slowk_trace_kwargs (dict): Keyword arguments passed to the trace of `slowk`.
        slowd_trace_kwargs (dict): Keyword arguments passed to the trace of `slowd`.
        fig (Figure or FigureWidget): Figure to add the traces to.
        **layout_kwargs: Keyword arguments passed to `fig.update_layout`.
```

Let's create a plot with two subplots: OHLC above, and %D and %K below. We will also change the 
style of both output lines from dashed to solid, and display a range between an oversold limit 
of 20 and an overbought limit of 80: 

```pycon
>>> fig = vbt.make_subplots(
...     rows=2, 
...     cols=1, 
...     shared_xaxes=True,  # (1)!
...     vertical_spacing=0.05)
>>> ohlc.vbt.ohlcv.plot(
...     add_trace_kwargs=dict(row=1, col=1),  # (2)!
...     fig=fig,
...     xaxis=dict(rangeslider_visible=False))  # (3)!
>>> stoch.plot(
...     limits=(20, 80),
...     add_trace_kwargs=dict(row=2, col=1),  # (4)!
...     slowk_trace_kwargs=dict(line=dict(dash=None)),  # (5)!
...     slowd_trace_kwargs=dict(line=dict(dash=None)),
...     fig=fig)
```

1. All time series have the same index
2. First subplot
3. Passed directly to `fig.update_layout`
4. Second subplot
5. Passed directly to `fig.add_trace`

![](/assets/images/indicators_stoch_subplots.svg)

## Pandas TA

[IndicatorFactory.from_pandas_ta](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_pandas_ta) 
can parse [Pandas TA](https://github.com/twopirllc/pandas-ta) indicators. Since Pandas TA indicators have
not metadata attached to each indicator, there is a method 
[IndicatorFactory.parse_pandas_ta_config](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.parse_pandas_ta_config) 
that reads the signature of an indicator function to get the input and parameter names and defaults, 
passes several dozens of rows of sample data to the function, and derives the number and 
names of the outputs.

!!! note
    If any indicator raises an error while parsing, try increasing the number of rows passed to the
    indicator function, for example, by passing `parse_kwargs=dict(test_index_len=150)`.

To get the list of all supported indicators:

```pycon
>>> vbt.IF.get_pandas_ta_indicators()  # (1)!
{'ABERRATION',
 'ACCBANDS',
 'AD',
 ...
 'XSIGNALS',
 'ZLMA',
 'ZSCORE'}
```

1. Using [IndicatorFactory.get_pandas_ta_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_pandas_ta_indicators) 

To get an indicator:

```pycon
>>> vbt.IF.from_pandas_ta('RSI')  # (1)!
vectorbtpro.indicators.factory.pandas_ta.RSI
```

1. Using [IndicatorFactory.from_pandas_ta](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_pandas_ta) 

Or, using a shortcut:

```pycon
>>> vbt.pandas_ta('RSI')
vectorbtpro.indicators.factory.pandas_ta.RSI
```

## TA

[IndicatorFactory.from_ta](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_ta) 
can parse [TA](https://github.com/bukosabino/ta) indicators. Similarly to Pandas TA, TA indicators
must be explicitly parsed to get the context of each indicator function. Since every indicator
is a class, there is a method [IndicatorFactory.parse_ta_config](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.parse_ta_config) 
that reads the signature, the docstring, and the attributes of the class to derive the input, parameter, 
and output names and defaults.

To get the list of all supported indicators:

```pycon
>>> vbt.IF.get_ta_indicators()  # (1)!
{'ADXIndicator',
 'AccDistIndexIndicator',
 'AroonIndicator',
 ...
 'VortexIndicator',
 'WMAIndicator',
 'WilliamsRIndicator'
```

1. Using [IndicatorFactory.get_ta_indicators](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.get_ta_indicators) 

To get an indicator:

```pycon
>>> vbt.IF.from_ta('RSIIndicator')  # (1)!
vectorbtpro.indicators.factory.ta.RSIIndicator
```

1. Using [IndicatorFactory.from_ta](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_ta) 

Or, using a shortcut:

```pycon
>>> vbt.ta('RSIIndicator')
vectorbtpro.indicators.factory.ta.RSIIndicator
```

## Expressions

Expressions are a brand-new way to define indicators of any complexity using regular strings.
The main advantage of expressions over custom and apply functions is that vectorbt can easily 
introspect the code of an indicator and inject a lot of useful automations.

Expressions are converted into full-blown indicators by a hybrid method 
[IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr).
Why hybrid? It's both a class and an instance method. We can call this method on an instance in case 
when we want to have a full control over the indicator's specification, and on a class in
case when we want the entire specification to be parsed for us. Let's try both approaches
while building an ATR indicator!

### Instance method

Here's a semi-automated implementation using the instance method:

```pycon
>>> expr = """
... tr0 = abs(high - low)
... tr1 = abs(high - fshift(close))
... tr2 = abs(low - fshift(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, n)
... tr, atr
... """
>>> ATR = vbt.IF(
...     class_name='ATR',
...     input_names=['high', 'low', 'close'],
...     param_names=['n'],
...     output_names=['tr', 'atr']
... ).from_expr(expr, n=14)

>>> atr = ATR.run(ohlc['high'], ohlc['low'], ohlc['close'])
>>> atr.atr
2019-12-31 00:00:00+00:00          NaN
2020-01-01 00:00:00+00:00          NaN
2020-01-02 00:00:00+00:00          NaN
2020-01-03 00:00:00+00:00          NaN
2020-01-04 00:00:00+00:00          NaN
...                                ...
2020-05-27 00:00:00+00:00    13.394434
2020-05-28 00:00:00+00:00    13.338482
2020-05-29 00:00:00+00:00    13.480809
2020-05-30 00:00:00+00:00    13.003231
2020-05-31 00:00:00+00:00    12.888624
Freq: D, Length: 153, dtype: float64
```

The expression `expr` is just a regular Python code without any extensions that gets evaluated using 
the Python's `eval` command. All function names are resolved by the parser prior to the evaluation.

### Class method

And here's a fully-automated implementation using the class method and annotations:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(@in_high - @in_low)
... tr1 = abs(@in_high - fshift(@in_close))
... tr2 = abs(@in_low - fshift(@in_close))
... @out_tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... @out_atr = wwm_mean_1d(@out_tr, @p_n)
... @out_tr, @out_atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14)
```

In the first example, we provided all the required information manually by constructing
a new instance of [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory).
The parser identified each of the input and parameter names in the expression, and replaced
them by the actual arrays. In the second example, we used annotations to give to the parser hints 
about the meaning of each variable. Whenever the parser finds a substring starting with `@`,
it knows that it has a special meaning for constructing a factory instance. For instance,
the prefixes `@_in`, `@_p`, and `@_out` indicate an input, parameter, and output respectively. 
The names appear in the order they appear in the expression (apart from OHLCV, where H comes always after O):

```pycon
>>> ATR.input_names
('high', 'low', 'close')
```

But the parser can even parse information that doesn't start with a special character.
For example, each of `open`, `high`, `low`, `close`, and `volume` are identified automatically, 
such that we don't need to provide annotations for them. They are referred to as magnet inputs, 
which are specified via the `magnet_inputs` argument. In case none of the outputs have annotations 
and the expression is a multi-line expression with the last line containing a tuple with valid 
variable names, we don't even have to provide annotations for the outputs. Also, as we saw above, 
the class name can be provided in the first line trailed by a colon:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - fshift(close))
... tr2 = abs(low - fshift(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, @p_n)
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14)
>>> ATR.input_names
('high', 'low', 'close')

>>> ATR.output_names
('tr', 'atr')
```

What about functions? The parser can identify functions by looking into various modules and packages.
In our example, `abs` and `nanmax` have been found in NumPy, while `wwm_mean_1d` has been 
found among generic Numba-compiled functions in [nb](/api/generic/nb/) (even without the `_nb` suffix). 
Look in the API of [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
to learn more. To avoid naming conflicts, we can still access the NumPy, Pandas, and vectorbt
modules via `np`, `pd`, and `vbt` respectively:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - vbt.nb.fshift_nb(close))
... tr2 = abs(low - vbt.nb.fshift_nb(close))
... tr = np.nanmax(np.column_stack((tr0, tr1, tr2)), axis=1)
... atr = vbt.nb.wwm_mean_1d_nb(tr, n)
... tr, atr
... """
```

### TA-Lib

Another automation touches TA-Lib indicators: vectorbt will replace any variable annotated
with `@talib` with an actual TA-Lib indicator function that can work on both one-dimensional and 
two-dimensional data!

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - fshift(close))
... tr2 = abs(low - fshift(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = @talib_ema(tr, 2 * n - 1)  # Wilder's EMA
... tr, atr
... """
```

### Context

So, how can we define our own functions and rules? Any additional keyword argument passed to
[IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
acts as a context for the evaluation and can replace a variable with the same name. Let's define our 
own function `shift_close` that acts as an alias for [fshift_nb](/api/generic/nb/#vectorbtpro.generic.nb.base.fshift_nb):

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - shift_close(close))
... tr2 = abs(low - shift_close(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, @p_n)
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14, shift_close=vbt.nb.fshift_nb)
```

We may have functions that depend on the evaluation context. In the example above,
we can make our `shift_close` accept the context and pull the number of periods to shift the closing price by
(just for the sake of example):

```pycon
>>> def shift_close(close, context):
...     return vbt.nb.fshift_nb(close, context.get('shift', 1))

>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - shift_close(close))
... tr2 = abs(low - shift_close(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, @p_n)
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14, shift_close=shift_close, shift=2)
```

The context will be automatically passed to the function once `context` has been recognized in its 
arguments. Moreover, we can make our `shift_close` to also pull the closing price itself.
Notice how `shift_close` takes no more arguments in the expression:

```pycon
>>> def shift_close(context):
...     return vbt.nb.fshift_nb(context['close'], context.get('shift', 1))

>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - shift_close())
... tr2 = abs(low - shift_close())
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, @p_n)
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14, shift_close=shift_close)
```

If we run this, we'll get an error stating that `close` was not found in the context. This is because
the input `close` is not "visible" in the expression, so it wasn't appended to the list of input names. 
To make any input, in-output, or parameter visible even without including it in the expression, 
we need to notify vectorbt that there is a function depending on it by using a dictionary called 
`func_mapping`, which takes functions and the magnet names they depend on:

```pycon
>>> func_mapping = dict(
...     shift_close=dict(
...         func=shift_close,
...         magnet_inputs=['close']
...     )
... )
>>> ATR = vbt.IF.from_expr(expr, n=14, func_mapping=func_mapping)
```

Since `shift_close` depends only on the context, we can instruct the parser to call it 
before the evaluation and only once, which would effectively cache it. For this, we need to use
`res_func_mapping` instead of `func_mapping`:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - shifted_close)
... tr2 = abs(low - shifted_close)
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, @p_n)
... tr, atr
... """
>>> res_func_mapping = dict(
...     shifted_close=dict(
...         func=shift_close,
...         magnet_inputs=['close']
...     )
... )
>>> ATR = vbt.IF.from_expr(expr, n=14, res_func_mapping=res_func_mapping)
```

Notice how `shifted_close` doesn't have parentheses anymore - it has become an array.

### Settings

But that's not all. How about overriding any information passed to 
[IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
but from within an expression? Wonder or not, this is also possible! We can define a dictionary
anywhere in the expression annotated with `@settings({...})`. The dictionary within the parentheses
will be evaluated with the Python `eval` command prior to the main evaluation and 
merged over the default settings of the factory method. 

Let's rewrite the [instance method](#instance-method) example solely using an expression:

```pycon
>>> expr = """
... @settings(dict(
...     factory_kwargs=dict(
...         class_name='ATR',
...         input_names=['high', 'low', 'close'],
...         param_names=['n'],
...         output_names=['tr', 'atr']
...     ),
...     n=14
... ))
...
... tr0 = abs(high - low)
... tr1 = abs(high - fshift(close))
... tr2 = abs(low - fshift(close))
... tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... atr = wwm_mean_1d(tr, n)
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr)
```

### Stacking

Remember that we can use arbitrary Python code inside our expressions, even other indicators. 
To simplify the usage of indicators, there is a convenient annotation `@res`, which takes 
the name of an indicator and creates an automatically resolved function out of it, just like 
`shifted_close` above. It makes this function an entry of `res_func_mapping` and puts the indicator's 
input, in-output, and parameter names into the entry's magnet lists, so we don't have to worry about 
passing the right information to the indicator - vectorbt does it for us!

Let's illustrate this awesomeness by defining basic SuperTrend bands:

```pycon
>>> expr = """
... SuperTrend[st]:
... avg_price = (high + low) / 2
... up = avg_price + @p_mult * @res_talib_atr
... down = avg_price - @p_mult * @res_talib_atr
... up, down
... """  # (1)!
>>> SuperTrend = vbt.IF.from_expr(expr, mult=3, atr_timeperiod=10)

>>> SuperTrend.input_names  # (2)!
('high', 'low', 'close')

>>> SuperTrend.param_names  # (3)!
('mult', 'atr_timeperiod')

>>> SuperTrend.output_names
('up', 'down')

>>> st = SuperTrend.run(ohlc['high'], ohlc['low'], ohlc['close'])
>>> st.up
2019-12-31 00:00:00+00:00           NaN
2020-01-01 00:00:00+00:00           NaN
2020-01-02 00:00:00+00:00           NaN
2020-01-03 00:00:00+00:00           NaN
2020-01-04 00:00:00+00:00           NaN
...                                 ...
2020-05-27 00:00:00+00:00    240.698333
2020-05-28 00:00:00+00:00    245.683118
2020-05-29 00:00:00+00:00    251.211675
2020-05-30 00:00:00+00:00    256.406868
2020-05-31 00:00:00+00:00    250.201819
Freq: D, Length: 153, dtype: float64
```

1. Short name of the indicator can be provided in the square brackets following the class name
2. The TA-Lib's ATR indicator depends on `high`, `low`, and `close`
3. Our indicator depends upon the multiplier `mult`, while the ATR also depends upon `timeperiod`. 

So, what happens if there are two indicators with overlapping inputs, parameters, or other arguments? 
Every argument apart from the inputs receives a prefix with the short name of the indicator
([IndicatorBase.short_name](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorBase.short_name)).
Under the hood, vectorbt traverses the signature of the indicator's `run` method, and looks
whether there is an argument with the same name in the context (and don't forget the prefix).

By default, the resolved function returns a raw output in form of one or more NumPy arrays. 
If the indicator has more than one output, we can use regular indexing to select a specific array, 
such as `@res_talib_macd[0]`. Let's disable raw outputs for ATR and query the `real` Pandas object on its 
indicator instance instead:

```pycon
>>> expr = """
... SuperTrend[st]:
... avg_price = (high + low) / 2
... up = avg_price + @p_mult * @res_talib_atr.real.values
... down = avg_price - @p_mult * @res_talib_atr.real.values
... up, down
... """
>>> SuperTrend = vbt.IF.from_expr(
...     expr, 
...     mult=3, 
...     atr_timeperiod=10, 
...     atr_kwargs=dict(return_raw=False))  # (1)!
```

1. If we printed `vbt.format_func(vbt.talib('ATR').run)`, we would see that any additional keyword argument
is passed as `**kwargs`, so we can specify `atr_kwargs` to target those variable arguments

### One-liners

There is nothing more satisfying than being able to define an indicator in one line :drooling_face:

```pycon
>>> AvgPrice = vbt.IF.from_expr("AvgPrice: @out_avg_price:(high + low) / 2")

>>> AvgPrice.run(ohlc['high'], ohlc['low']).avg_price
2019-12-31 00:00:00+00:00    100.496714
2020-01-01 00:00:00+00:00    100.216791
2020-01-02 00:00:00+00:00     92.735610
2020-01-03 00:00:00+00:00     90.353978
2020-01-04 00:00:00+00:00     91.676538
...                                 ...
2020-05-27 00:00:00+00:00    200.426358
2020-05-28 00:00:00+00:00    205.655007
2020-05-29 00:00:00+00:00    210.587055
2020-05-30 00:00:00+00:00    217.806298
2020-05-31 00:00:00+00:00    212.041686
Freq: D, Length: 153, dtype: float64
```

Notice how the output annotation `@out` isn't bound to any variable anymore but is written similarly
to the class name - with a trailing colon following the expression of the output.
If there are multiple outputs, their output expressions must be separated by a comma.
Here's a single-line expression for basic SuperTrend bands with multiple outputs:

```pycon
>>> SuperTrend = vbt.IF.from_expr(
...     "SuperTrend[st]: @out_up:@res_avg_price + @p_mult * @res_talib_atr, "
...     "@out_down:@res_avg_price - @p_mult * @res_talib_atr",  # (1)!
...     avg_price=AvgPrice,
...     atr_timeperiod=10, 
...     mult=3)
>>> st = SuperTrend.run(ohlc['high'], ohlc['low'], ohlc['close'])

>>> fig = ohlc.vbt.ohlcv.plot()
>>> st.up.rename('Upper').vbt.plot(fig=fig)
>>> st.down.rename('Lower').vbt.plot(fig=fig)
```

1. Split the expression for better readability

![](/assets/images/indicators_supertrend.svg)

### Using Pandas

Like many other factory methods, [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
passes inputs and in-outputs as two-dimensional NumPy arrays. We can enable the `keep_pd` flag to work 
on Pandas objects. Let's run our ATR indicator using Pandas alone:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(high - low)
... tr1 = abs(high - close.shift())
... tr2 = abs(low - close.shift())
... tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
... atr = tr.ewm(alpha=1 / @p_n, adjust=False, min_periods=@p_n).mean()
... tr, atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14, keep_pd=True)
>>> atr = ATR.run(ohlc['high'], ohlc['low'], ohlc['close'])
>>> atr.atr
2019-12-31 00:00:00+00:00          NaN
2020-01-01 00:00:00+00:00          NaN
2020-01-02 00:00:00+00:00          NaN
2020-01-03 00:00:00+00:00          NaN
2020-01-04 00:00:00+00:00          NaN
                               ...    
2020-05-27 00:00:00+00:00    13.394434
2020-05-28 00:00:00+00:00    13.338482
2020-05-29 00:00:00+00:00    13.480809
2020-05-30 00:00:00+00:00    13.003231
2020-05-31 00:00:00+00:00    12.888624
Freq: D, Length: 153, dtype: float64
```

!!! note
    In contrast to the previous NumPy-only expressions, this expression won't work for 
    multiple columns of input data.

For simpler expressions, we can instruct the parser to use 
[pandas.eval](https://pandas.pydata.org/docs/reference/api/pandas.eval.html) instead of the Python's `eval`.
This brings multi-threading and other performance advantages for big inputs since `pd.eval` 
switches to [NumExpr](https://github.com/pydata/numexpr) by default:

```pycon
>>> AvgPrice = vbt.IF.from_expr(
...     "AvgPrice: @out_avg_price:(high + low) / 2",
...     use_pd_eval=True
... )

>>> AvgPrice.run(ohlc['high'], ohlc['low']).avg_price
2019-12-31 00:00:00+00:00    100.496714
2020-01-01 00:00:00+00:00    100.216791
2020-01-02 00:00:00+00:00     92.735610
2020-01-03 00:00:00+00:00     90.353978
2020-01-04 00:00:00+00:00     91.676538
                                ...    
2020-05-27 00:00:00+00:00    200.426358
2020-05-28 00:00:00+00:00    205.655007
2020-05-29 00:00:00+00:00    210.587055
2020-05-30 00:00:00+00:00    217.806298
2020-05-31 00:00:00+00:00    212.041686
Freq: D, Length: 153, dtype: float64
```

### Debugging

To see the expression after parsing all the annotations, set `return_clean_expr` to True:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(@in_high - @in_low)
... tr1 = abs(@in_high - fshift(@in_close))
... tr2 = abs(@in_low - fshift(@in_close))
... @out_tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... @out_atr = wwm_mean_1d(@out_tr, @p_n)
... @out_tr, @out_atr
... """
>>> print(vbt.IF.from_expr(expr, n=14, return_clean_expr=True))
tr0 = abs(__in_high - __in_low)
tr1 = abs(__in_high - fshift(__in_close))
tr2 = abs(__in_low - fshift(__in_close))
__out_tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
__out_atr = wwm_mean_1d(__out_tr, __p_n)
__out_tr, __out_atr
```

Additionally, just like in a regular Python code, we can place `print` statements to
explore the state at each execution step:

```pycon
>>> expr = """
... ATR:
... tr0 = abs(@in_high - @in_low)
... print('tr0: ', tr0.shape)
... tr1 = abs(@in_high - fshift(@in_close))
... print('tr1: ', tr1.shape)
... tr2 = abs(@in_low - fshift(@in_close))
... print('tr2: ', tr2.shape)
... @out_tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
... print('tr: ', @out_tr.shape)
... @out_atr = wwm_mean_1d(@out_tr, @p_n)
... print('atr: ', @out_atr.shape)
... @out_tr, @out_atr
... """
>>> ATR = vbt.IF.from_expr(expr, n=14)
>>> atr = ATR.run(ohlc['high'], ohlc['low'], ohlc['close'])
tr0:  (153, 1)
tr1:  (153, 1)
tr2:  (153, 1)
tr:  (153,)
atr:  (153,)
```

## WorldQuant's Alphas

[IndicatorFactory.from_wqa101](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_wqa101)
uses the expression parser to parse and execute [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf).
Each alpha expression is defined in [wqa101_expr_config](/api/indicators/expr/#vectorbtpro.indicators.expr.wqa101_expr_config),
while most functions and resolved functions used in alpha expressions are defined in
[expr_func_config](/api/indicators/expr/#vectorbtpro.indicators.expr.expr_func_config) and 
[expr_res_func_config](/api/indicators/expr/#vectorbtpro.indicators.expr.expr_res_func_config) respectively.

To get an indicator:

```pycon
>>> WQA53 = vbt.IF.from_wqa101(53)
>>> wqa53 = WQA53.run(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
>>> wqa53.out
2019-12-31 00:00:00+00:00         NaN
2020-01-01 00:00:00+00:00         NaN
2020-01-02 00:00:00+00:00         NaN
2020-01-03 00:00:00+00:00         NaN
2020-01-04 00:00:00+00:00         NaN
                               ...   
2020-05-27 00:00:00+00:00    0.193719
2020-05-28 00:00:00+00:00   -0.858778
2020-05-29 00:00:00+00:00    0.452096
2020-05-30 00:00:00+00:00    0.376475
2020-05-31 00:00:00+00:00   -0.539368
Freq: D, Length: 153, dtype: float64
```

Or, using a shortcut:

```pycon
>>> vbt.wqa101(53)
vectorbtpro.indicators.factory.wqa101.WQA53
```

Replicating an alpha indicator is quite easy: look up its expression in the config and 
pass to [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr):

```pycon
>>> WQA53 = vbt.IF.from_expr("-delta(((close - low) - (high - close)) / (close - low), 9)")
>>> wqa53 = WQA53.run(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
>>> wqa53.out
2019-12-31 00:00:00+00:00         NaN
2020-01-01 00:00:00+00:00         NaN
2020-01-02 00:00:00+00:00         NaN
2020-01-03 00:00:00+00:00         NaN
2020-01-04 00:00:00+00:00         NaN
                               ...   
2020-05-27 00:00:00+00:00    0.193719
2020-05-28 00:00:00+00:00   -0.858778
2020-05-29 00:00:00+00:00    0.452096
2020-05-30 00:00:00+00:00    0.376475
2020-05-31 00:00:00+00:00   -0.539368
Freq: D, Length: 153, dtype: float64
```