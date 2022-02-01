---
title: Design
---

# Design

An indicator is a pipeline that does the following:

* Accepts input arrays (for example, the opening and closing price)
* Accepts parameters either in a scalar or array format (for example, window size)
* Accepts other relevant arguments and keyword arguments
* Broadcasts input arrays against each other or some shape
* Broadcasts parameters against each other to form a fixed set of parameter combinations
* For each parameter combination, performs calculation on the input arrays to produce output arrays 
of the same shape (for example, rolling average)
* Concatenates output arrays of all parameter combinations along columns
* Converts the results back into the Pandas format

Let's manually create an indicator that takes two time series, calculates their normalized moving
averages, and returns the difference of both. We'll test different shapes as well 
as parameter combinations to take advantage of broadcasting:

```pycon
>>> import vectorbtpro as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime, timedelta

>>> def mov_avg_crossover(ts1, ts2, w1, w2):
...     ts1, ts2 = vbt.broadcast(ts1, ts2)  # (1)!
...
...     w1, w2 = vbt.broadcast(  # (2)!
...         vbt.to_1d_array(w1), 
...         vbt.to_1d_array(w2))
...
...     ts1_mas = []
...     for w in w1:
...         ts1_mas.append(ts1.vbt.rolling_mean(w) / ts1)  # (3)!
...     ts2_mas = []
...     for w in w2:
...         ts2_mas.append(ts2.vbt.rolling_mean(w) / ts2)
...
...     ts1_ma = pd.concat(ts1_mas, axis=1)  # (4)!
...     ts2_ma = pd.concat(ts2_mas, axis=1)
...
...     ts1_ma.columns = vbt.combine_indexes((  # (5)!
...         pd.Index(w1, name="ts1_window"), 
...         ts1.columns))
...     ts2_ma.columns = vbt.combine_indexes((
...         pd.Index(w2, name="ts2_window"), 
...         ts2.columns))
...
...     return ts1_ma.vbt - ts2_ma  # (6)!

>>> def generate_index(n):  # (6)!
...     return [
...         datetime(2020, 1, 1) + timedelta(days=i) 
...         for i in range(n)
...     ]

>>> ts1 = pd.Series([1, 2, 3, 4, 5, 6, 7], index=generate_index(7))
>>> ts2 = pd.DataFrame({
...     'a': [5, 4, 3, 2, 3, 4, 5],
...     'b': [2, 3, 4, 5, 4, 3, 2]
... }, index=generate_index(7))
>>> w1 = 2
>>> w2 = [3, 4]

>>> mov_avg_crossover(ts1, ts2, w1, w2)
ts1_window                                       2
ts2_window                   3                   4
                   a         b         a         b
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.500000  0.083333       NaN       NaN
2020-01-04 -0.625000  0.075000 -0.875000  0.175000
2020-01-05  0.011111 -0.183333 -0.100000 -0.100000
2020-01-06  0.166667 -0.416667  0.166667 -0.416667
2020-01-07  0.128571 -0.571429  0.228571 -0.821429
```

1. Both time series become Pandas objects with the same shape - `(7, 2)`
2. Window `w1` becomes `[2, 2]` and window `w2` becomes `[3, 4]`. This builds two parameter
combinations: `(2, 3)` and `(2, 4)`.
3. Calculate the normalized moving average based on each of the windows
4. Concatenate the DataFrames along the column axis
5. Create a new column hierarchy with the window values on top
6. Calculate the difference of both concatenated DataFrames
7. Convenient function to generate a datetime-like index

Neat! We just created a pretty flexible pipeline that takes arbitrary input and parameter combinations.
The end result of this pipeline is a DataFrame where each column corresponds to a single window
combination applied on a single column in both `ts1` and `ts2`. But is this pipeline user-friendly? :thinking:
Having to deal with broadcasting, output concatenation, and the column hierarchy makes this code no different
from a regular Pandas code.

The pipeline above can be well standardized, which is done by 
[run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline).
It conveniently prepares inputs, parameters, and columns, while the calculation and output 
concatenation have to be performed by the user using `custom_func`. 
Let's rewrite the above example a bit:

```pycon
>>> from vectorbtpro.indicators.factory import run_pipeline

>>> def custom_func(ts1, ts2, w1, w2):
...     ts1_mas = []
...     for w in w1:
...         ts1_mas.append(vbt.nb.rolling_mean_nb(ts1, w) / ts1)  # (1)!
...     ts2_mas = []
...     for w in w2:
...         ts2_mas.append(vbt.nb.rolling_mean_nb(ts2, w) / ts2)
...
...     ts1_ma = np.column_stack(ts1_mas)  # (2)!
...     ts2_ma = np.column_stack(ts2_mas)
...
...     return ts1_ma - ts2_ma  # (3)!

>>> outputs = run_pipeline(
...     num_ret_outputs=1,
...     custom_func=custom_func,
...     inputs=dict(ts1=ts1, ts2=ts2),
...     params=dict(w1=w1, w2=w2)
... )
>>> outputs
(<vectorbtpro.base.wrapping.ArrayWrapper at 0x7fb188993160>,
 [array([[1, 1],
         [2, 2],
         [3, 3],
         [4, 4],
         [5, 5],
         [6, 6],
         [7, 7]]),
  array([[5, 2],
         [4, 3],
         [3, 4],
         [2, 5],
         [3, 4],
         [4, 3],
         [5, 2]])],
 array([0, 1, 0, 1]),
 [],
 [array([[        nan,         nan,         nan,         nan],
         [        nan,         nan,         nan,         nan],
         [-0.5       ,  0.08333333,         nan,         nan],
         [-0.625     ,  0.075     , -0.875     ,  0.175     ],
         [ 0.01111111, -0.18333333, -0.1       , -0.1       ],
         [ 0.16666667, -0.41666667,  0.16666667, -0.41666667],
         [ 0.12857143, -0.57142857,  0.22857143, -0.82142857]])],
 [[2, 2], [3, 4]],
 [Int64Index([2, 2, 2, 2], dtype='int64'),
  Int64Index([3, 3, 4, 4], dtype='int64')],
 [])
```

1. Using [rolling_mean_nb](/api/generic/nb/#vectorbtpro.generic.nb.rolling_mean_nb)
2. Concatenate the NumPy arrays along the column axis
3. Calculate the difference of both concatenated NumPy arrays

We produced much less code and did the entire calculation using NumPy and Numba alone - a big win!
But what is this monstrous output? 

This raw output is really meant to be used by vectorbt, not by the user themselves - it contains useful
metadata for working the indicator. Additionally, if you look into the source of this function, you will 
notice that it accepts a ton of arguments. Great complexity enables great flexibility: each argument is 
targeted at configuring a specific step of the pipeline. But don't worry: we won't use this function directly.

## Factory

Instead, we will use [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory),
which simplifies the usage of [run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline)
by providing a unified interface and various automations. Let's wrap our `custom_func` using the factory:

```pycon
>>> MADiff = vbt.IF(
...     class_name='MADiff',
...     input_names=['ts1', 'ts2'],
...     param_names=['w1', 'w2'],
...     output_names=['diff'],
... ).with_custom_func(custom_func)

>>> madiff = MADiff.run(ts1, ts2, w1, w2)
>>> madiff.diff
madiff_w1                                        2
madiff_w2                    3                   4
                   a         b         a         b
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.500000  0.083333       NaN       NaN
2020-01-04 -0.625000  0.075000 -0.875000  0.175000
2020-01-05  0.011111 -0.183333 -0.100000 -0.100000
2020-01-06  0.166667 -0.416667  0.166667 -0.416667
2020-01-07  0.128571 -0.571429  0.228571 -0.821429
```

!!! hint
    `vbt.IF` is a shortcut for [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory).

As you see, [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory) 
took the specification for our indicator and created an entire Python class that knows how to communicate
with [run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline) and manipulate
and format its results. In particular, it attached the class method `MADiff.run` that looks exactly like 
`custom_func` but prepares and forwards all arguments to 
[run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline) under the hood. 
Whenever we call the `run` method, it sets up and returns an instance of `MADiff` with all the input 
and output data.

You might ask: *"Why doesn't the factory create a function instead of a class? Having an indicator
function would be more intuitive!"* If you read through [Building blocks](/documentation/building-blocks),
you would already be familiar with the class [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable),
which is the go-to class for analyzing data. The indicator class created by the factory is a subclass of 
[Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable), so we not only 
have access to the output, but also to many methods for analyzing this output! For example,
the factory automatically attaches `crossed_above`, `cross_below`, `stats`, and many other 
methods for each input and output that appears in the indicator:

```pycon
>>> madiff.diff_stats(column=(2, 3, 'a'))
Start        2020-01-01 00:00:00
End          2020-01-07 00:00:00
Period           7 days 00:00:00
Count                          5
Mean                    -0.16373
Std                     0.371153
Min                       -0.625
Median                  0.011111
Max                     0.166667
Min Index    2020-01-04 00:00:00
Max Index    2020-01-06 00:00:00
Name: (2, 3, a), dtype: object
```

## Workflow

The main purpose of [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory)
is to create a stand-alone indicator class that has a `run` method for running the indicator. 
For this, it needs to know what inputs, parameters, and outputs to expect. This information can be 
passed in form of `input_names`, `param_names`, and other arguments to the constructor:

```pycon
>>> MADiff_factory = vbt.IF(
...     class_name='MADiff',
...     input_names=['ts1', 'ts2'],
...     param_names=['w1', 'w2'],
...     output_names=['diff'],
... )
>>> MADiff_factory.Indicator
vectorbtpro.indicators.factory.MADiff
```

Upon the initialization, it creates the skeleton of our indicator class of type
[IndicatorBase](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorBase), accessible via 
[IndicatorFactory.Indicator](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.Indicator).
Even though the factory has created the constructor of this class and attached various
properties and methods for working with it, we can't run the indicator:

```pycon
>>> MADiff_factory.Indicator.run()
NotImplementedError: 
```

This is because we haven't provided it with the calculation function yet. To do this,
there are multiple methods starting with the prefix `with_`. The base method all other
methods are based upon is [IndicatorFactory.with_custom_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_custom_func)
(which we used above) - it overrides the abstract `run` method to execute the indicator using 
[run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline)
and returns a ready-to-use indicator class:

```pycon
>>> MADiff = MADiff_factory.with_custom_func(custom_func)
>>> MADiff
vectorbtpro.indicators.factory.MADiff
```

The calculation function has been attached successfully, we can now run this indicator!

## Factory methods

Factory methods come in two different flavors: instance and class methods. The instance methods with 
the prefix `with_`, such as [IndicatorFactory.with_custom_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_custom_func),
require instantiation of the indicator factory. That is, we have to do `vbt.IF(...)` and provide the 
required information manually as we did with `MADiff`. The class methods with the prefix `from_`, 
such as [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr),
can parse the required information (semi-)automatically.

### From custom function

The method [IndicatorFactory.with_custom_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_custom_func)
takes a so-called "custom function", which is the most flexible way to define an indicator. 
But with great power comes great responsibility: it's up to the user to iterate through parameters,
handle caching, and concatenate columns for each parameter (usually by 
[apply_and_concat](/api/base/combining/#vectorbtpro.base.combining.apply_and_concat)). 
Also, we must ensure that each output array has an appropriate number of columns, which is the 
number of columns in the input arrays multiplied by the number of parameter combinations.
Additionally, the custom function receives commands passed by the pipeline, and it's the task
of the user to properly execute those commands.

For example, if our custom function needs the index and the columns along with the NumPy arrays,
we can instruct the pipeline to pass the wrapper, which is done by setting `pass_wrapper=True`
in `with_custom_func`. This as well as all other arguments are forwarded directly to 
[run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline),
which takes care of communicating with our custom function.

### From apply function

The method [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_apply_func)
simplifies indicator development a lot: it creates `custom_func` that handles caching, 
iteration over parameters, and output concatenation, and passes this function to 
[IndicatorFactory.with_custom_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_custom_func).
Our part is writing a so-called "apply function", which accepts a single parameter combination and does 
the calculation. It then automatically concatenates the resulting outputs along the column axis.

!!! note
    An apply function has mostly the same signature as a custom function, but the parameters
    are single values as opposed to multiple values.

Let's implement our indicator using an apply function:

```pycon
>>> def apply_func(ts1, ts2, w1, w2):
...     ts1_ma = vbt.nb.rolling_mean_nb(ts1, w1) / ts1
...     ts2_ma = vbt.nb.rolling_mean_nb(ts2, w2) / ts2
...     return ts1_ma - ts2_ma

>>> MADiff = vbt.IF(
...     class_name='MADiff',
...     input_names=['ts1', 'ts2'],
...     param_names=['w1', 'w2'],
...     output_names=['diff'],
... ).with_apply_func(apply_func)

>>> madiff = MADiff.run(ts1, ts2, w1, w2)
>>> madiff.diff
madiff_w1                                        2
madiff_w2                    3                   4
                   a         b         a         b
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.500000  0.083333       NaN       NaN
2020-01-04 -0.625000  0.075000 -0.875000  0.175000
2020-01-05  0.011111 -0.183333 -0.100000 -0.100000
2020-01-06  0.166667 -0.416667  0.166667 -0.416667
2020-01-07  0.128571 -0.571429  0.228571 -0.821429
```

That's it! Under the hood, our code created a custom function that iterates over both
parameter combinations and calls `apply_func` on each one. If we printed `ts1`, `ts2`, `w1`, and `w2`, 
we would see that `ts1` and `ts2` are the same, while `w1` and `w2` are now single values.
This way, we can entirely abstract ourselves from the number of parameter combinations
and work with a single set of parameters at a time.

Another advantage of this method is that apply functions are natural inhabitants of vectorbt :monkey:
and we can use most regular and Numba-compiled functions that take two-dimensional NumPy arrays
directly as apply functions! Let's illustrate this by building an indicator for the rolling covariance:

```pycon
>>> RollCov = vbt.IF(
...     class_name='RollCov',
...     input_names=['ts1', 'ts2'],
...     param_names=['w'],
...     output_names=['rollcov'],
... ).with_apply_func(vbt.nb.rolling_cov_nb)

>>> rollcov = RollCov.run(ts1, ts2, [2, 3])
>>> rollcov.rollcov
rollcov_w            2                   3
               a     b         a         b
2020-01-01   NaN   NaN       NaN       NaN
2020-01-02 -0.25  0.25       NaN       NaN
2020-01-03 -0.25  0.25 -0.666667  0.666667
2020-01-04 -0.25  0.25 -0.666667  0.666667
2020-01-05  0.25 -0.25  0.000000  0.000000
2020-01-06  0.25 -0.25  0.666667 -0.666667
2020-01-07  0.25 -0.25  0.666667 -0.666667
```

Here, the both input arrays and the window parameter were passed directly to 
[rolling_cov_nb](/api/generic/nb/#vectorbtpro.generic.nb.rolling_cov_nb).

#### Custom iteration

We can easily emulate `apply_func` using `custom_func` and 
[apply_and_concat](/api/base/combining/#vectorbtpro.base.combining.apply_and_concat),
for example, if we need the index of the current iteration and/or want to have access to all parameter
combinations:

```pycon
>>> from vectorbtpro.base.combining import apply_and_concat

>>> def apply_func(i, ts1, ts2, w):  # (1)!
...     return vbt.nb.rolling_cov_nb(ts1, ts2, w[i])

>>> def custom_func(ts1, ts2, w):
...     return apply_and_concat(len(w), apply_func, ts1, ts2, w)  # (2)!

>>> RollCov = vbt.IF(
...     class_name='RollCov',
...     input_names=['ts1', 'ts2'],
...     param_names=['w'],
...     output_names=['rollcov'],
... ).with_custom_func(custom_func)
```

1. In contrast to our previous `apply_func`, an apply function used in 
[apply_and_concat](/api/base/combining/#vectorbtpro.base.combining.apply_and_concat) must take
the index of the iteration and select the perameters manually using this index
2. [apply_and_concat](/api/base/combining/#vectorbtpro.base.combining.apply_and_concat)
requires the number of iterations, which is simply the length of any parameter array

The same using [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.with_apply_func)
and `select_params=False`:

```pycon
>>> RollCov = vbt.IF(
...     class_name='RollCov',
...     input_names=['ts1', 'ts2'],
...     param_names=['w'],
...     output_names=['rollcov'],
... ).with_apply_func(apply_func, select_params=False)
```

#### Debugging

Sometimes, it's not that clear which arguments are being passed to `apply_func`.
Debugging in this scenario is usually easy: just replace your apply function with a generic 
apply function that takes variables arguments, and print those.

```pycon
>>> def apply_func(*args, **kwargs):
...     for i, arg in enumerate(args):
...         print("arg {}: {}".format(i, type(arg)))
...     for k, v in kwargs.items():
...         print("kwarg {}: {}".format(k, type(v)))
...     raise NotImplementedError

>>> RollCov = vbt.IF(
...     class_name='RollCov',
...     input_names=['ts1', 'ts2'],
...     param_names=['w'],
...     output_names=['rollcov'],
... ).with_apply_func(apply_func, select_params=False)

>>> try:
...     RollCov.run(ts1, ts2, [2, 3], some_arg="some_value")
... except:
...     pass
arg 0: <class 'int'>
arg 1: <class 'numpy.ndarray'>
arg 2: <class 'numpy.ndarray'>
arg 3: <class 'list'>
kwarg some_arg: <class 'str'>
```

### From parsing

Parsers are the most convenient way to build indicator classes. For instance, there are dedicated 
parser methods for third-party technical analysis packages that can derive the specification of each 
indicator in an (semi-)automated way. In addition, there is a powerful expression parser to avoid writing 
complex Python functions for simpler indicators. Let's *express* our indicator as an expression:

```pycon
>>> MADiff = vbt.IF.from_expr(
...     "rolling_mean(@in_ts1, @p_w1) / @in_ts1 - rolling_mean(@in_ts2, @p_w2) / @in_ts2",
...     factory_kwargs=dict(class_name="MADiff")  # (1)!
... )
>>> madiff = MADiff.run(ts1, ts2, w1, w2)
>>> madiff.out
madiff_w1                                        2
madiff_w2                    3                   4
                   a         b         a         b
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.500000  0.083333       NaN       NaN
2020-01-04 -0.625000  0.075000 -0.875000  0.175000
2020-01-05  0.011111 -0.183333 -0.100000 -0.100000
2020-01-06  0.166667 -0.416667  0.166667 -0.416667
2020-01-07  0.128571 -0.571429  0.228571 -0.821429
```

1. We can still override any information passed to the factory class

Notice how we didn't have to call `vbt.IF(...)`? [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
is a class method that parses `input_names` and other information from the expression and 
creates a factory instance using solely this information. Crazy how we compressed our first
implementation with `mov_avg_crossover` to just this while enjoying all the perks, right?

## Run methods

Once we built our indicator class, it's time to run it. The main method for executing an indicator 
is the class method [IndicatorBase.run](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorBase.run), 
which accepts positional and keyword arguments based on the specification provided to the 
[IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory).
These arguments include input arrays, in-place output arrays, and parameters. Any additional arguments
are forwarded down to [run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline), 
which can either use them to set up the pipeline, or forward them further down to the custom function
and then, if provided, the apply function.

To see what arguments the `run` method accepts, use the Python's `help` function:

```pycon
>>> help(MADiff.run)
Help on method run:

run(ts1, ts2, w1, w2, short_name='madiff', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
    Run `MADiff` indicator.
    
    * Inputs: `ts1`, `ts2`
    * Parameters: `w1`, `w2`
    * Outputs: `diff`
    
    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.
    
    Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
```

We see that `MADiff.run` takes two input time series `ts1` and `ts2`, two parameters `w1` and `w2`, 
and produces a single output time series `diff`. Upon calling the class method, it runs the indicator
and returns a new instance of `MADiff` with all the data being ready for analysis. In particular, 
we can access the output as a regular instance attribute `MADiff.diff`.

The second method for running indicators is [IndicatorBase.run_combs](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorBase.run_combs), 
which takes the same inputs as the method above, but computes all combinations of the passed parameters 
based on a combinatorial function and returns **multiple** indicator instances that can be combined with 
each other. This is useful to compare multiple indicators of the **same type** but different
parameters, such as for testing a moving average crossover, which involves two 
[MA](/api/indicators/custom/#vectorbtpro.indicators.custom.MA) instances applied on the same time series:

```pycon
>>> ts = pd.Series([3, 2, 1, 2, 3])
>>> fast_ma, slow_ma = vbt.MA.run_combs(
...     ts, [2, 3, 4], 
...     short_names=['fast_ma', 'slow_ma'])
>>> fast_ma.ma_crossed_above(slow_ma)
fast_ma_window             2      3
slow_ma_window      3      4      4
0               False  False  False
1               False  False  False
2               False  False  False
3               False  False  False
4                True   True  False
```

In the example above, [MA.run_combs](/api/indicators/custom/#vectorbtpro.indicators.custom.MA.run_combs)
generated the combinations of `window` using [itertools.combinations](https://docs.python.org/3/library/itertools.html#itertools.combinations) 
and `r=2`. The first set of window combinations was passed to the first instance, the second set
to the second instance. The above example can be easily replicated using the `run` method alone:

```pycon
>>> import itertools

>>> windows = [2, 3, 4]
>>> fast_windows, slow_windows = zip(*itertools.combinations(windows, 2))
>>> fast_ma = vbt.MA.run(ts, fast_windows, short_name='fast_ma')
>>> slow_ma = vbt.MA.run(ts, slow_windows, short_name='slow_ma')
>>> fast_ma.ma_crossed_above(slow_ma)
fast_ma_window             2      3
slow_ma_window      3      4      4
0               False  False  False
1               False  False  False
2               False  False  False
3               False  False  False
4                True   True  False
```

The main advantage of a single `run_combs` call over multiple `run` calls is that it doesn't 
need to re-compute each combination thanks to smart caching.

!!! note
    `run_combs` should be only used for combining multiple indicators. To test multiple
    parameter combinations, use `run` and provide parameters as lists.