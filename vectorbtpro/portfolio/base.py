# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for modeling portfolio and measuring its performance.

Provides the class `vectorbtpro.portfolio.base.Portfolio` for modeling portfolio performance
and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbtpro.portfolio.nb` for most computations and record classes based on
`vectorbtpro.records.base.Records` for evaluating events such as orders, logs, trades, positions, and drawdowns.

The job of the `Portfolio` class is to create a series of positions allocated 
against a cash component, produce an equity curve, incorporate basic transaction costs
and produce a set of statistics about its performance. In particular, it outputs
position/profit metrics and drawdown information.

Run for the examples below:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime
>>> import talib
>>> from numba import njit

>>> import vectorbtpro as vbt
>>> from vectorbtpro.utils.colors import adjust_opacity
>>> from vectorbtpro.utils.enum_ import map_enum_fields
```

## Workflow

`Portfolio` class does quite a few things to simulate our strategy.

**Preparation** phase (in the particular class method):

* Receives a set of inputs, such as signal arrays and other parameters
* Resolves parameter defaults by searching for them in the global settings
* Brings input arrays to a single shape
* Does some basic validation of inputs and converts Pandas objects to NumPy arrays
* Passes everything to a Numba-compiled simulation function

**Simulation** phase (in the particular simulation function using Numba):

* The simulation function traverses the broadcasted shape element by element, row by row (time dimension),
    column by column (asset dimension)
* For each asset and timestamp (= element):
    * Gets all available information related to this element and executes the logic
    * Generates an order or skips the element altogether
    * If an order has been issued, processes the order and fills/ignores/rejects it
    * If the order has been filled, registers the result by appending it to the order records
    * Updates the current state such as the cash and asset balances

**Construction** phase (in the particular class method):

* Receives the returned order records and initializes a new `Portfolio` object

**Analysis** phase (in the `Portfolio` object)

* Offers a broad range of risk & performance metrics based on order records

## Simulation modes

There are three main simulation modes.

### From orders

`Portfolio.from_orders` is the most straightforward and the fastest out of all simulation modes.

An order is a simple instruction that contains size, price, fees, and other information
(see `vectorbtpro.portfolio.enums.Order` for details about what information a typical order requires).
Instead of creating a `vectorbtpro.portfolio.enums.Order` tuple for each asset and timestamp (which may
waste a lot of memory) and appending it to a (potentially huge) list for processing, `Portfolio.from_orders`
takes each of those information pieces as an array, broadcasts them against each other, and creates a
`vectorbtpro.portfolio.enums.Order` tuple out of each element for us.

Thanks to broadcasting, we can pass any of the information as a 2-dim array, as a 1-dim array
per column or row, and as a constant. And we don't even need to provide every piece of information -
vectorbt fills the missing data with default constants, without wasting memory.

Here's an example:

```pycon
>>> size = pd.Series([1, -1, 1, -1])  # per row
>>> price = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})  # per element
>>> direction = ['longonly', 'shortonly']  # per column
>>> fees = 0.01  # per frame

>>> pf = vbt.Portfolio.from_orders(price, size, direction=direction, fees=fees)
>>> pf.orders.records_readable
   Order Id Column  Timestamp  Size  Price  Fees  Side
0         0      a          0   1.0    1.0  0.01   Buy
1         1      a          1   1.0    2.0  0.02  Sell
2         2      a          2   1.0    3.0  0.03   Buy
3         3      a          3   1.0    4.0  0.04  Sell
4         0      b          0   1.0    4.0  0.04  Sell
5         1      b          1   1.0    3.0  0.03   Buy
6         2      b          2   1.0    2.0  0.02  Sell
7         3      b          3   1.0    1.0  0.01   Buy
```

This method is particularly useful in situations where you don't need any further logic
apart from filling orders at predefined timestamps. If you want to issue orders depending
upon the previous performance, the current state, or other custom conditions, head over to
`Portfolio.from_signals` or `Portfolio.from_order_func`.

### From signals

`Portfolio.from_signals` is centered around signals. It adds an abstraction layer on top of `Portfolio.from_orders`
to automate some signaling processes. For example, by default, it won't let us execute another entry signal
if we are already in the position. It also implements stop loss and take profit orders for exiting positions.
Nevertheless, this method behaves similarly to `Portfolio.from_orders` and accepts most of its arguments;
in fact, by setting `accumulate=True`, it behaves quite similarly to `Portfolio.from_orders`.

Let's replicate the example above using signals:

```pycon
>>> entries = pd.Series([True, False, True, False])
>>> exits = pd.Series([False, True, False, True])

>>> pf = vbt.Portfolio.from_signals(
...     price,
...     entries, exits,
...     size=1, direction=direction, fees=fees)
>>> pf.orders.records_readable
   Order Id Column  Timestamp  Size  Price  Fees  Side
0         0      a          0   1.0    1.0  0.01   Buy
1         1      a          1   1.0    2.0  0.02  Sell
2         2      a          2   1.0    3.0  0.03   Buy
3         3      a          3   1.0    4.0  0.04  Sell
4         0      b          0   1.0    4.0  0.04  Sell
5         1      b          1   1.0    3.0  0.03   Buy
6         2      b          2   1.0    2.0  0.02  Sell
7         3      b          3   1.0    1.0  0.01   Buy
```

In a nutshell: this method automates some procedures that otherwise would be only possible by using
`Portfolio.from_order_func` while following the same broadcasting principles as `Portfolio.from_orders` -
the best of both worlds, given you can express your strategy as a sequence of signals. But as soon as
your strategy requires any signal to depend upon more complex conditions or to generate multiple orders at once,
it's best to run your custom signaling logic using `Portfolio.from_order_func`.

### From order function

`Portfolio.from_order_func` is the most powerful form of simulation. Instead of pulling information
from predefined arrays, it lets us define an arbitrary logic through callbacks. There are multiple
kinds of callbacks, each called at some point while the simulation function traverses the shape.
For example, apart from the main callback that returns an order (`order_func_nb`), there is a callback
that does preprocessing on the entire group of columns at once. For more details on the general procedure
and the callback zoo, see `vectorbtpro.portfolio.nb.from_order_func.simulate_nb`.

Let's replicate our example using an order function:

```pycon
>>> @njit
>>> def order_func_nb(c, size, direction, fees):
...     return vbt.pf_nb.order_nb(
...         price=c.close[c.i, c.col],
...         size=size[c.i],
...         direction=direction[c.col],
...         fees=fees
... )

>>> direction_num = map_enum_fields(direction, Direction)
>>> pf = vbt.Portfolio.from_order_func(
...     price,
...     order_func_nb,
...     np.asarray(size), np.asarray(direction_num), fees
... )
>>> pf.orders.records_readable
   Order Id Column  Timestamp  Size  Price  Fees  Side
0         0      a          0   1.0    1.0  0.01   Buy
1         1      a          1   1.0    2.0  0.02  Sell
2         2      a          2   1.0    3.0  0.03   Buy
3         3      a          3   1.0    4.0  0.04  Sell
4         0      b          0   1.0    4.0  0.04  Sell
5         1      b          1   1.0    3.0  0.03   Buy
6         2      b          2   1.0    2.0  0.02  Sell
7         3      b          3   1.0    1.0  0.01   Buy
```

There is an even more flexible version available - `vectorbtpro.portfolio.nb.from_order_func.flex_simulate_nb`
(activated by passing `flexible=True` to `Portfolio.from_order_func`) - that allows creating multiple
orders per symbol and bar.

This method has many advantages:

* Realistic simulation as it follows the event-driven approach - less risk of exposure to the look-ahead bias
* Provides a lot of useful information during the runtime, such as the current position's PnL
* Enables putting all logic including custom indicators into a single place, and running it as the data
 comes in, in a memory-friendly manner

But there are drawbacks too:

* Doesn't broadcast arrays - needs to be done by the user prior to the execution
* Requires at least a basic knowledge of NumPy and Numba
* Requires at least an intermediate knowledge of both to optimize for efficiency

## Example

To showcase the features of `Portfolio`, run the following example: it checks candlestick data of 6 major
cryptocurrencies in 2020 against every single pattern found in TA-Lib, and translates them into orders.

```pycon
>>> # Fetch price history
>>> symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'BCH-USD', 'LTC-USD']
>>> start = '2020-01-01 UTC'  # crypto is UTC
>>> end = '2020-09-01 UTC'
>>> # OHLCV by column
>>> ohlcv = vbt.YFData.fetch(symbols, start=start, end=end).concat()
```

[=100% "100%"]{: .candystripe}

```pycon
>>> ohlcv['Open']
symbol                          BTC-USD     ETH-USD   XRP-USD    BNB-USD  \\
Date
2020-01-01 00:00:00+00:00   7194.892090  129.630661  0.192912  13.730962
2020-01-02 00:00:00+00:00   7202.551270  130.820038  0.192708  13.698126
2020-01-03 00:00:00+00:00   6984.428711  127.411263  0.187948  13.035329
...                                 ...         ...       ...        ...
2020-08-29 00:00:00+00:00  11541.054688  395.687592  0.272009  23.134024
2020-08-30 00:00:00+00:00  11508.713867  399.616699  0.274568  23.009060
2020-08-31 00:00:00+00:00  11713.306641  428.509003  0.283065  23.647858

symbol                        BCH-USD    LTC-USD
Date
2020-01-01 00:00:00+00:00  204.671295  41.326534
2020-01-02 00:00:00+00:00  204.354538  42.018085
2020-01-03 00:00:00+00:00  196.007690  39.863129
...                               ...        ...
2020-08-29 00:00:00+00:00  269.112976  57.438873
2020-08-30 00:00:00+00:00  268.842865  57.207737
2020-08-31 00:00:00+00:00  279.280426  62.844059

[243 rows x 6 columns]

>>> # Run every single pattern recognition indicator and combine the results
>>> result = vbt.pd_acc.empty_like(ohlcv['Open'], fill_value=0.)
>>> for pattern in talib.get_function_groups()['Pattern Recognition']:
...     PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
...     pr = PRecognizer.run(ohlcv['Open'], ohlcv['High'], ohlcv['Low'], ohlcv['Close'])
...     result = result + pr.integer

>>> # Don't look into the future
>>> result = result.vbt.fshift(1)

>>> # Treat each number as order value in USD
>>> size = result / ohlcv['Open']

>>> # Simulate portfolio
>>> pf = vbt.Portfolio.from_orders(
...     ohlcv['Close'], size, price=ohlcv['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001)

>>> # Visualize portfolio value
>>> pf.value.vbt.plot().show()
```

![](/assets/images/api/portfolio_value.svg)

## Broadcasting

`Portfolio` is very flexible towards inputs:

* Accepts both Series and DataFrames as inputs
* Broadcasts inputs to the same shape using `vectorbtpro.base.reshaping.broadcast`
* Many inputs (such as `fees`) can be passed as a single value, value per column/row, or as a matrix
* Implements flexible indexing wherever possible to save memory

## Defaults

If you look at the arguments of each class method, you will notice that most of them default to None.
`None` has a special meaning in vectorbt: it's a command to pull the default value from the global
config with settings `vectorbtpro._settings.portfolio`. For example, the default size used in
`Portfolio.from_signals` and `Portfolio.from_orders` is `np.inf`:

```pycon
>>> vbt.settings.portfolio['size']
inf
```

## Attributes

Once a portfolio is built, it gives us the possibility to assess its performance from
various angles. There are three main types of portfolio attributes:

* time series in form of a Series/DataFrame (such as running cash balance),
* time series reduced per column/group in form of a scalar/Series (such as total return), and
* records in form of a structured NumPy array (such as order records).

Time series take a lot of memory, especially when hyperparameter optimization is involved.
To avoid wasting resources, they are not computed during the simulation but reconstructed
from order records and other data (see `vectorbtpro.portfolio.enums.SimulationOutput`). This way,
any attribute is only computed once the user actually needs it.

Since most attributes of a portfolio must first be reconstructed, they have a getter method.
For example, to reconstruct the cash balance at each time step, we call `Portfolio.get_cash`.
Additionally, each attribute has a shortcut property (`Portfolio.cash` in our example)
that calls the getter method with default arguments.

```pycon
>>> pf.cash.equals(pf.get_cash())
True
```

There are two main advantages of shortcut properties:

1) They are cacheable
2) They can return in-output arrays pre-computed during the simulation

All of this makes them very fast to access. Moreover, attributes that need to call
other attributes can utilize their shortcut properties by calling `Portfolio.resolve_shortcut_attr`,
which calls the respective shortcut property whenever default arguments are passed.

## Grouping

One of the key features of `Portfolio` is the ability to group columns. Groups can be specified by
`group_by`, which can be anything from positions or names of column levels, to a NumPy array with
actual groups. Groups can be formed to share capital between columns (make sure to pass `cash_sharing=True`)
or to compute metrics for a combined portfolio of multiple independent columns.

For example, let's divide our portfolio into two groups sharing the same cash balance:

```pycon
>>> # Simulate combined portfolio
>>> group_by = pd.Index([
...     'first', 'first', 'first',
...     'second', 'second', 'second'
... ], name='group')
>>> comb_pf = vbt.Portfolio.from_orders(
...     ohlcv['Close'], size, price=ohlcv['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001,
...     group_by=group_by, cash_sharing=True)

>>> # Get total profit per group
>>> comb_pf.total_profit
group
first     22353.207762
second     7634.297901
Name: total_profit, dtype: float64
```

Not only can we analyze each group, but also each column in the group:

```pycon
>>> # Get total profit per column
>>> comb_pf.get_total_profit(group_by=False)
symbol
BTC-USD     5233.981995
ETH-USD    13814.978843
XRP-USD     3304.246924
BNB-USD     4725.737791
BCH-USD     -255.652597
LTC-USD     3164.212707
Name: total_profit, dtype: float64
```

In the same way, we can introduce new grouping to the method itself:

```pycon
>>> # Get total profit per group
>>> pf.get_total_profit(group_by=group_by)
group
first     22353.207762
second     7634.297901
Name: total_profit, dtype: float64
```

!!! note
    If cash sharing is enabled, grouping can be disabled but cannot be modified.

## Indexing

Like any other class subclassing `vectorbtpro.base.wrapping.Wrapping`, we can do pandas indexing
on a `Portfolio` instance, which forwards indexing operation to each object with columns:

```pycon
>>> pf['BTC-USD']
<vectorbtpro.portfolio.base.Portfolio at 0x7f7812364f98>

>>> pf['BTC-USD'].total_profit
5233.981994880156
```

Grouped portfolio is indexed by group:

```pycon
>>> comb_pf['first']
<vectorbtpro.portfolio.base.Portfolio at 0x7f7811177400>

>>> comb_pf['first'].total_profit
22353.207761869122
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `pf.iloc[0]` instead of `pf.iloc[:, 0]`
    to get the first column.

    Indexing behavior depends solely upon `vectorbtpro.base.wrapping.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns. You can pass wrapper arguments in `broadcast_kwargs`.

## Logging

To collect more information on how a specific order was processed or to be able to track the whole
simulation from the beginning to the end, we can turn on logging:

```pycon
>>> # Simulate portfolio with logging
>>> pf = vbt.Portfolio.from_orders(
...     ohlcv['Close'], size, price=ohlcv['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001, log=True)

>>> pf.logs.records
       id  group  col  idx  open  high  low        close  cash    position  \\
0       0      0    0    0   NaN   NaN  NaN  7200.174316   inf    0.000000
1       1      0    0    1   NaN   NaN  NaN  6985.470215   inf    0.000000
2       2      0    0    2   NaN   NaN  NaN  7344.884277   inf    0.000000
...   ...    ...  ...  ...   ...   ...  ...          ...   ...         ...
1455  240      5    5  240   NaN   NaN  NaN    57.291733   inf  268.907681
1456  241      5    5  241   NaN   NaN  NaN    62.725342   inf  272.389644
1457  242      5    5  242   NaN   NaN  NaN    61.113796   inf  274.137659

      ...  new_free_cash  new_val_price  new_value  res_size    res_price  \\
0     ...            inf    7194.892090        inf       NaN          NaN
1     ...            inf    7202.551270        inf       NaN          NaN
2     ...            inf    6984.428711        inf       NaN          NaN
...   ...            ...            ...        ...       ...          ...
1455  ...            inf      57.438873        inf  3.481962    57.496312
1456  ...            inf      57.207737        inf  1.748015    57.264945
1457  ...            inf      62.844059        inf  7.956202    62.906903

      res_fees  res_side  res_status  res_status_info  order_id
0          NaN        -1           1                0        -1
1          NaN        -1           1                5        -1
2          NaN        -1           1                5        -1
...        ...       ...         ...              ...       ...
1455    0.2002         0           0               -1       181
1456    0.1001         0           0               -1       182
1457    0.5005         0           0               -1       183

[1458 rows x 43 columns]
```

Just as orders, logs are also records and thus can be easily analyzed:

```pycon
>>> pf.logs.res_status.value_counts()
symbol   BTC-USD  ETH-USD  XRP-USD  BNB-USD  BCH-USD  LTC-USD
Filled       183      172      176      178      176      184
Ignored       60       71       67       65       67       59
```

Logging can also be turned on just for one order, row, or column, since as many other
variables it's specified per order and can broadcast automatically.

!!! note
    Logging can slow down simulation.

## Caching

`Portfolio` heavily relies upon caching. Most shortcut properties are wrapped with a
cacheable decorator: reduced time series and records are automatically cached
using `vectorbtpro.utils.decorators.cached_property`, while time-series are not cached
automatically but are cacheable using `vectorbtpro.utils.decorators.cacheable_property`,
meaning you must explicitly turn them on.

!!! note
    Shortcut properties are only triggered once default arguments are passed to a method.
    Explicitly disabling/enabling grouping will not trigger them so the whole call hierarchy
    cannot utilize caching anymore. To still utilize caching, we need to create a new
    portfolio object with disabled/enabled grouping using `new_pf = pf.replace(group_by=my_group_by)`.

Caching can be disabled globally via `vectorbtpro._settings.caching`.
Alternatively, we can precisely point at attributes and methods that should or shouldn't
be cached. For example, we can blacklist the entire `Portfolio` class:

```pycon
>>> vbt.Portfolio.get_ca_setup().disable_caching()
```

Or a single instance of `Portfolio`:

```pycon
>>> pf.get_ca_setup().disable_caching()
```

See `vectorbtpro.registries.ca_registry` for more details on caching.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `Portfolio.replace` method and pass changes as keyword arguments.

## Performance and memory

### Caching attributes manually

If you're running out of memory when working with large arrays, disable caching.

Also make sure to store most important time series manually if you're planning to re-use them.
For example, if you're interested in Sharpe ratio or other metrics based on returns,
run and save `Portfolio.returns` to a variable, delete the portfolio object, and then use the
`vectorbtpro.returns.accessors.ReturnsAccessor` to analyze them. Do not use methods akin to
`Portfolio.sharpe_ratio` because they will re-calculate returns each time (unless you turned
on caching for time series).

```pycon
>>> returns_acc = pf.returns_acc
>>> del pf
>>> returns_acc.sharpe_ratio()
symbol
BTC-USD    1.617912
ETH-USD    2.568341
XRP-USD    1.381798
BNB-USD    1.525383
BCH-USD   -0.013760
LTC-USD    0.934991
Name: sharpe_ratio, dtype: float64
```

Many methods such as `Portfolio.get_returns` are both instance and class methods. Running the instance method
will trigger a waterfall of computations, such as getting cash flow, asset flow, etc. Some of these
attributes are calculated more than once. For example, `Portfolio.get_net_exposure` must compute
`Portfolio.get_gross_exposure` for long and short positions. Each call of `Portfolio.get_gross_exposure`
must recalculate the cash series from scratch if caching for them is disabled. To avoid this, use class methods:

```pycon
>>> free_cash = pf.free_cash  # reuse wherever possible
>>> long_exposure = vbt.Portfolio.get_gross_exposure(
...     asset_value=pf.get_asset_value(direction='longonly'),
...     free_cash=free_cash,
...     wrapper=pf.wrapper
... )
>>> short_exposure = vbt.Portfolio.get_gross_exposure(
...     asset_value=pf.get_asset_value(direction='shortonly'),
...     free_cash=free_cash,
...     wrapper=pf.wrapper
... )
>>> del free_cash  # release memory
>>> net_exposure = vbt.Portfolio.get_net_exposure(
...     long_exposure=long_exposure,
...     short_exposure=short_exposure,
...     wrapper=pf.wrapper
... )
>>> del long_exposure  # release memory
>>> del short_exposure  # release memory
```

### Pre-calculating attributes

Instead of computing memory and CPU-expensive attributes such as `Portfolio.returns` retroactively,
we can pre-calculate them during the simulation using `Portfolio.from_order_func` and its callbacks.
For this, we need to pass `in_outputs` argument with an empty floating array, fill it in
`post_segment_func_nb`, and `Portfolio` will automatically use it as long as we don't change grouping:

```pycon
>>> pf_baseline = vbt.Portfolio.from_orders(
...     ohlcv['Close'], size, price=ohlcv['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001, freq='d')
>>> pf_baseline.sharpe_ratio
symbol
BTC-USD    1.617912
ETH-USD    2.568341
XRP-USD    1.381798
BNB-USD    1.525383
BCH-USD   -0.013760
LTC-USD    0.934991
Name: sharpe_ratio, dtype: float64

>>> @njit
... def order_func_nb(c, size, price, fees, slippage):
...     return vbt.pf_nb.order_nb(
...         size=vbt.pf_nb.select_nb(c, size),
...         price=vbt.pf_nb.select_nb(c, price),
...         fees=vbt.pf_nb.select_nb(c, fees),
...         slippage=vbt.pf_nb.select_nb(c, slippage),
...     )

>>> @njit
... def post_segment_func_nb(c):
...     if c.cash_sharing:
...         c.in_outputs.returns[c.i, c.group] = c.last_return[c.group]
...     else:
...         for col in range(c.from_col, c.to_col):
...             c.in_outputs.returns[c.i, col] = c.last_return[col]

>>> pf = vbt.Portfolio.from_order_func(
...     ohlcv['Close'],
...     order_func_nb,
...     vbt.to_2d_array(size),
...     vbt.to_2d_array(ohlcv['Open']),
...     vbt.to_2d_array(0.001),
...     vbt.to_2d_array(0.001),
...     post_segment_func_nb=post_segment_func_nb,
...     in_outputs=dict(returns=vbt.RepEval("np.empty_like(close, dtype=np.float_)")),
...     init_cash=pf_baseline.init_cash,
...     freq='d'
... )
>>> pf.sharpe_ratio
symbol
BTC-USD    1.617912
ETH-USD    2.568341
XRP-USD    1.381798
BNB-USD    1.525383
BCH-USD   -0.013760
LTC-USD    0.934991
Name: sharpe_ratio, dtype: float64
```

To make sure that we used the pre-calculated array:

```pycon
>>> vbt.settings.caching['disable'] = True

>>> # Reconstructed
>>> %timeit pf.get_returns()
5.82 ms ± 58.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> # Pre-computed
>>> %timeit pf.returns
70.1 µs ± 219 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

The only drawback of this approach is that you cannot use `init_cash='auto'` or `init_cash='autoalign'`
because then, during the simulation, the portfolio value is `np.inf` and the returns are `np.nan`.

You should also take care of grouping the pre-computed array during the simulation.
For example, running the above function with grouping but without cash sharing will throw an error.
To provide a hint to vectorbt that the array should only be used when cash sharing is enabled,
add the suffix '_cs' to the name of the array (see `Portfolio.in_outputs_indexing_func` on supported suffixes).

### Chunking simulation

As most Numba-compiled functions in vectorbt, simulation procedure can also be chunked and run in parallel.
For this, use the `chunked` argument (see `vectorbtpro.utils.chunking.resolve_chunked_option`).
For example, let's simulate 1 million orders 1) without chunking, 2) sequentially, and 2) concurrently using Dask:

```pycon
>>> size = np.full((1000, 1000), 1.)
>>> size[1::2] = -1
>>> %timeit vbt.Portfolio.from_orders(1, size)
90.1 ms ± 8.15 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit vbt.Portfolio.from_orders(1, size, chunked=True)
110 ms ± 10 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit vbt.Portfolio.from_orders(1, size, chunked='dask')
43.6 ms ± 2.39 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Since the chunking procedure is applied on the Numba-compiled function itself (see the source
of the particular function), the fastest execution engine is always a multi-threaded one.
Executing chunks sequentially does not result in a speedup and is pretty useless in this scenario
because there is always an overhead of splitting and distributing the arguments and merging the results.

Chunking happens (semi-)automatically by splitting each argument into chunks of columns.
It does not break groups, thus chunking is safe on any portfolio regardless of its grouping.

!!! warning
    Additional arguments such as `signal_args` in `Portfolio.from_signals` are not split
    automatically and require providing a specification, otherwise they are passed as-is.
    See examples under `vectorbtpro.utils.chunking.chunked`.

### Chunking everything

Simulation in chunks improves performance but doesn't help with memory: every array needs to be loaded
into memory in order to be split. A better idea is to keep one chunk in memory at a time. For example, we
can build a chunkable pipeline that loads a chunk of data, performs the simulation on that chunk,
calculates all relevant metrics, and merges the results across all chunks.

Let's create a pipeline that tests various window combinations of a moving average crossover and
concatenates their total returns:

```pycon
>>> @vbt.chunked(
...     size=vbt.LenSizer(arg_query='fast_windows'),
...     arg_take_spec=dict(
...         price=None,
...         fast_windows=vbt.ChunkSlicer(),
...         slow_windows=vbt.ChunkSlicer()
...     ),
...     merge_func=lambda x: pd.concat(x).vbt.sort_index()
... )
... def pipeline(price, fast_windows, slow_windows):
...     fast_ma = vbt.MA.run(price, fast_windows, short_name='fast')
...     slow_ma = vbt.MA.run(price, slow_windows, short_name='slow')
...     entries = fast_ma.ma_crossed_above(slow_ma)
...     exits = fast_ma.ma_crossed_below(slow_ma)
...     pf = vbt.Portfolio.from_signals(price, entries, exits)
...     return pf.total_return

>>> price = vbt.YFData.fetch(['BTC-USD', 'ETH-USD']).get('Close')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> pipeline(price, [10, 10, 10], [20, 30, 50])
fast_window  slow_window  symbol
10           20           BTC-USD      172.663535
                          ETH-USD     2213.427388
             30           BTC-USD      175.853073
                          ETH-USD    16197.543067
             50           BTC-USD      122.635872
                          ETH-USD     2116.661012
Name: total_return, dtype: float64
```

Let's find out how the function splits data into 2 chunks (this won't trigger the computation):

```pycon
>>> chunk_meta, funcs_args = pipeline(
...     price, [10, 10, 10], [20, 30, 50],
...     _n_chunks=2, _return_raw_chunks=True
... )
>>> chunk_meta
[ChunkMeta(uuid='d87cee40-11ac-4e96-8e1d-ec1b613263a0', idx=0, start=0, end=2, indices=None),
 ChunkMeta(uuid='3eb4189e-3d70-4763-9cc2-35f3c91bc105', idx=1, start=2, end=3, indices=None)]

>>> list(funcs_args)
[(<function __main__.pipeline(price, fast_windows, slow_windows)>,
  (symbol                          BTC-USD      ETH-USD
   Date
   2014-09-17 00:00:00+00:00    457.334015          NaN
   2014-09-18 00:00:00+00:00    424.440002          NaN
   2014-09-19 00:00:00+00:00    394.795990          NaN
   ...                                 ...          ...
   2021-12-15 00:00:00+00:00  48896.722656  4018.388672
   2021-12-16 00:00:00+00:00  47665.425781  3962.469727
   2021-12-18 00:00:00+00:00  46725.929688  3940.733398

   [2645 rows x 2 columns],                                         << price (unchanged)
   [10, 10],                                                        << fast_windows (1st chunk)
   [20, 30]),                                                       << slow_windows (1st chunk)
  {}),
 (<function __main__.pipeline(price, fast_windows, slow_windows)>,
  (symbol                          BTC-USD      ETH-USD
   Date
   2014-09-17 00:00:00+00:00    457.334015          NaN
   2014-09-18 00:00:00+00:00    424.440002          NaN
   2014-09-19 00:00:00+00:00    394.795990          NaN
   ...                                 ...          ...
   2021-12-15 00:00:00+00:00  48896.722656  4018.388672
   2021-12-16 00:00:00+00:00  47665.425781  3962.469727
   2021-12-18 00:00:00+00:00  46725.929688  3940.733398

   [2645 rows x 2 columns],
   [10],                                                            << price (unchanged)
   [50]),                                                           << fast_windows (2nd chunk)
  {})]                                                              << slow_windows (2nd chunk)
```

We see that the function correctly chunked `fast_windows` and `slow_windows` and left the data as it is.

## Saving and loading

Like any other class subclassing `vectorbtpro.utils.pickling.Pickleable`, we can save a `Portfolio`
instance to the disk with `Portfolio.save` and load it with `Portfolio.load`:

```pycon
>>> pf = vbt.Portfolio.from_orders(
...     ohlcv['Close'], size, price=ohlcv['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001, freq='d')
>>> pf.sharpe_ratio
symbol
BTC-USD    1.617912
ETH-USD    2.568341
XRP-USD    1.381798
BNB-USD    1.525383
BCH-USD   -0.013760
LTC-USD    0.934991
Name: sharpe_ratio, dtype: float64

>>> pf.save('my_pf')
>>> pf = vbt.Portfolio.load('my_pf')
>>> pf.sharpe_ratio
symbol
BTC-USD    1.617912
ETH-USD    2.568341
XRP-USD    1.381798
BNB-USD    1.525383
BCH-USD   -0.013760
LTC-USD    0.934991
Name: sharpe_ratio, dtype: float64
```

!!! note
    Save files won't include neither cached results nor global defaults. For example,
    passing `fillna_close` as None will also use None when the portfolio is loaded from disk.
    Make sure to either pass all arguments explicitly or to also save the `vectorbtpro._settings.portfolio` config.

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Portfolio.metrics`.

Let's simulate a portfolio with two columns:

```pycon
>>> close = vbt.YFData.fetch(
...     "BTC-USD",
...     start='2020-01-01 UTC',
...     end='2020-09-01 UTC'
... ).get('Close')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> pf = vbt.Portfolio.from_random_signals(close, n=[10, 20], seed=42)
>>> pf.wrapper.columns
Int64Index([10, 20], dtype='int64', name='rand_n')
```

### Column, group, and tag selection

To return the statistics for a particular column/group, use the `column` argument:

```pycon
>>> pf.stats(column=10)
UserWarning: Metric 'sharpe_ratio' requires frequency to be set
UserWarning: Metric 'calmar_ratio' requires frequency to be set
UserWarning: Metric 'omega_ratio' requires frequency to be set
UserWarning: Metric 'sortino_ratio' requires frequency to be set
UserWarning: Couldn't parse the frequency of index. Pass it as `freq` or define it globally under `settings.wrapping`.

Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                              243
Start Value                                       100.0
End Value                                    139.876426
Total Return [%]                              39.876426
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                                12.7421
Max Drawdown Duration                             109.0
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration                     7.571429
Avg Losing Trade Duration                     12.333333
Profit Factor                                  2.941353
Expectancy                                     3.987643
Name: 10, dtype: object
```

If vectorbt couldn't parse the frequency of `close`:

1) it won't return any duration in time units,
2) it won't return any metric that requires annualization, and
3) it will throw a bunch of warnings (you can silence those by passing `silence_warnings=True`)

We can provide the frequency as part of the settings dict:

```pycon
>>> pf.stats(column=10, settings=dict(freq='d'))
UserWarning: Changing the frequency will create a copy of this object.
Consider setting the frequency upon object creation to re-use existing cache.

Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    139.876426
Total Return [%]                              39.876426
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                                12.7421
Max Drawdown Duration                 109 days 00:00:00
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration    7 days 13:42:51.428571428
Avg Losing Trade Duration              12 days 08:00:00
Profit Factor                                  2.941353
Expectancy                                     3.987643
Sharpe Ratio                                   1.515967
Calmar Ratio                                   5.117177
Omega Ratio                                    1.495807
Sortino Ratio                                  2.624107
Name: 10, dtype: object
```

But in this case, our portfolio will be copied to set the new frequency and we wouldn't be
able to re-use its cached attributes. Let's define the frequency upon the simulation instead:

```pycon
>>> pf = vbt.Portfolio.from_random_signals(close, n=[10, 20], seed=42, freq='d')
```

We can change the grouping of the portfolio on the fly. Let's form a single group:

```pycon
>>> pf.stats(group_by=True)
Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       200.0
End Value                                    344.850193
Total Return [%]                              72.425097
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                               9.849795
Max Drawdown Duration                  60 days 00:00:00
Total Trades                                         30
Total Closed Trades                                  30
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                  66.666667
Best Trade [%]                                20.650796
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.074987
Avg Losing Trade [%]                          -2.783814
Avg Winning Trade Duration              9 days 01:12:00
Avg Losing Trade Duration               5 days 19:12:00
Profit Factor                                  4.730329
Expectancy                                      4.82834
Sharpe Ratio                                   2.309773
Calmar Ratio                                  12.782774
Omega Ratio                                    1.557084
Sortino Ratio                                  3.945651
Name: group, dtype: object
```

We can see how the initial cash has changed from $100 to $200, indicating that both columns now
contribute to the performance.

### Aggregation

If the portfolio consists of multiple columns/groups and no column/group has been selected,
each metric is aggregated across all columns/groups based on `agg_func`, which is `np.mean` by default.

```pycon
>>> pf.stats()
UserWarning: Object has multiple columns. Aggregating using <function mean at 0x7fc77152bb70>.
Pass column to select a single column/group.

Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    172.425097
Total Return [%]                              72.425097
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                              13.935332
Max Drawdown Duration                  97 days 00:00:00
Total Trades                                       15.0
Total Closed Trades                                15.0
Total Open Trades                                   0.0
Open Trade PnL                                      0.0
Win Rate [%]                                       67.5
Best Trade [%]                                17.977121
Worst Trade [%]                               -7.432006
Avg Winning Trade [%]                          7.143562
Avg Losing Trade [%]                          -3.400855
Avg Winning Trade Duration    8 days 17:00:39.560439560
Avg Losing Trade Duration               7 days 16:00:00
Profit Factor                                  4.840401
Expectancy                                     4.618165
Sharpe Ratio                                   1.936813
Calmar Ratio                                   8.923935
Omega Ratio                                     1.55757
Sortino Ratio                                  3.440712
Name: agg_stats, dtype: object
```

Here, the Sortino ratio of 2.624107 (column=10) and 4.257318 (column=20) lead to the avarage of 3.440712.

We can also return a DataFrame with statistics per column/group by passing `agg_func=None`:

```pycon
>>> pf.stats(agg_func=None)
                             Start                       End   Period  ...  Sortino Ratio
randnx_n                                                               ...
10       2020-01-01 00:00:00+00:00 2020-08-31 00:00:00+00:00 243 days  ...       2.624107
20       2020-01-01 00:00:00+00:00 2020-08-31 00:00:00+00:00 243 days  ...       4.257318

[2 rows x 28 columns]
```

### Metric selection

To select metrics, use the `metrics` argument (see `Portfolio.metrics` for supported metrics):

```pycon
>>> pf.stats(metrics=['sharpe_ratio', 'sortino_ratio'], column=10)
Sharpe Ratio     1.515967
Sortino Ratio    2.624107
Name: 10, dtype: float64
```

We can also select specific tags (see any metric from `Portfolio.metrics` that has the `tag` key):

```pycon
>>> pf.stats(column=10, tags=['trades'])
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration    7 days 13:42:51.428571428
Avg Losing Trade Duration              12 days 08:00:00
Profit Factor                                  2.941353
Expectancy                                     3.987643
Name: 10, dtype: object
```

Or provide a boolean expression:

```pycon
>>> pf.stats(column=10, tags='trades and open and not closed')
Total Open Trades    0.0
Open Trade PnL       0.0
Name: 10, dtype: float64
```

The reason why we included "not closed" along with "open" is because some metrics such as the win rate
have both tags attached since they are based upon both open and closed trades/positions
(to see this, pass `settings=dict(incl_open=True)` and `tags='trades and open'`).

### Passing parameters

We can use `settings` to pass parameters used across multiple metrics.
For example, let's pass required and risk-free return to all return metrics:

```pycon
>>> pf.stats(column=10, settings=dict(required_return=0.1, risk_free=0.01))
Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    139.876426
Total Return [%]                              39.876426
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                                12.7421
Max Drawdown Duration                 109 days 00:00:00
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration    7 days 13:42:51.428571428
Avg Losing Trade Duration              12 days 08:00:00
Profit Factor                                  2.941353
Expectancy                                     3.987643
Sharpe Ratio                                  -8.119004  << here
Calmar Ratio                                   5.117177  << here
Omega Ratio                                    0.264555  << here
Sortino Ratio                                -18.734479  << here
Name: 10, dtype: object
```

Passing any argument inside of `settings` either overrides an existing default, or acts as
an optional argument that is passed to the calculation function upon resolution (see below).
Both `required_return` and `risk_free` can be found in the signature of the 4 ratio methods,
so vectorbt knows exactly it has to pass them.

Let's imagine that the signature of `vectorbtpro.returns.accessors.ReturnsAccessor.sharpe_ratio`
doesn't list those arguments: vectorbt would simply call this method without passing those two arguments.
In such case, we have two options:

1) Set parameters globally using `settings` and set `pass_{arg}=True` individually using `metric_settings`:

```pycon
>>> pf.stats(
...     column=10,
...     settings=dict(required_return=0.1, risk_free=0.01),
...     metric_settings=dict(
...         sharpe_ratio=dict(pass_risk_free=True),
...         omega_ratio=dict(pass_required_return=True, pass_risk_free=True),
...         sortino_ratio=dict(pass_required_return=True)
...     )
... )
```

2) Set parameters individually using `metric_settings`:

```pycon
>>> pf.stats(
...     column=10,
...     metric_settings=dict(
...         sharpe_ratio=dict(risk_free=0.01),
...         omega_ratio=dict(required_return=0.1, risk_free=0.01),
...         sortino_ratio=dict(required_return=0.1)
...     )
... )
```

### Custom metrics

To calculate a custom metric, we need to provide at least two things: short name and a settings
dict with the title and calculation function (see arguments in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`):

```pycon
>>> max_winning_streak = (
...     'max_winning_streak',
...     dict(
...         title='Max Winning Streak',
...         calc_func=lambda trades: trades.winning_streak.max(),
...         resolve_trades=True
...     )
... )
>>> pf.stats(metrics=max_winning_streak, column=10)
Max Winning Streak    3.0
Name: 10, dtype: float64
```

You might wonder how vectorbt knows which arguments to pass to `calc_func`?
In the example above, the calculation function expects two arguments: `trades` and `group_by`.
To automatically pass any of the them, vectorbt searches for each in the current settings.
As `trades` cannot be found, it either throws an error or tries to resolve this argument if
`resolve_{arg}=True` was passed. Argument resolution is the process of searching for property/method with
the same name (also with prefix `get_`) in the attributes of the current portfolio, automatically passing the
current settings such as `group_by` if they are present in the method's signature
(a similar resolution procedure), and calling the method/property. The result of the resolution
process is then passed as `arg` (or `trades` in our example).

Here's an example without resolution of arguments:

```pycon
>>> max_winning_streak = (
...     'max_winning_streak',
...     dict(
...         title='Max Winning Streak',
...         calc_func=lambda self, group_by:
...         self.get_trades(group_by=group_by).winning_streak.max()
...     )
... )
>>> pf.stats(metrics=max_winning_streak, column=10)
Max Winning Streak    3.0
Name: 10, dtype: float64
```

And here's an example without resolution of the calculation function:

```pycon
>>> max_winning_streak = (
...     'max_winning_streak',
...     dict(
...         title='Max Winning Streak',
...         calc_func=lambda self, settings:
...         self.get_trades(group_by=settings['group_by']).winning_streak.max(),
...         resolve_calc_func=False
...     )
... )
>>> pf.stats(metrics=max_winning_streak, column=10)
Max Winning Streak    3.0
Name: 10, dtype: float64
```

Since `max_winning_streak` method can be expressed as a path from this portfolio, we can simply write:

```pycon
>>> max_winning_streak = (
...     'max_winning_streak',
...     dict(
...         title='Max Winning Streak',
...         calc_func='trades.winning_streak.max'
...     )
... )
```

In this case, we don't have to pass `resolve_trades=True` any more as vectorbt does it automatically.
Another advantage is that vectorbt can access the signature of the last method in the path
(`vectorbtpro.records.mapped_array.MappedArray.max` in our case) and resolve its arguments.

To switch between entry trades, exit trades, and positions, use the `trades_type` setting.
Additionally, you can pass `incl_open=True` to also include open trades.

```pycon
>>> pf.stats(column=10, settings=dict(trades_type='positions', incl_open=True))
Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    139.876426
Total Return [%]                              39.876426
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                                12.7421
Max Drawdown Duration                 109 days 00:00:00
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration    7 days 13:42:51.428571428
Avg Losing Trade Duration              12 days 08:00:00
Profit Factor                                  2.941353
Expectancy                                     3.987643
Sharpe Ratio                                   1.515967
Calmar Ratio                                   5.117177
Omega Ratio                                    1.495807
Sortino Ratio                                  2.624107
Name: 10, dtype: object
```

Any default metric setting or even global setting can be overridden by the user using metric-specific
keyword arguments. Here, we override the global aggregation function for `max_dd_duration`:

```pycon
>>> pf.stats(agg_func=lambda sr: sr.mean(),
...     metric_settings=dict(
...         max_dd_duration=dict(agg_func=lambda sr: sr.max())
...     )
... )
UserWarning: Object has multiple columns. Aggregating using <function <lambda> at 0x7fbf6e77b268>.
Pass column to select a single column/group.

Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    172.425097
Total Return [%]                              72.425097
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                              13.935332
Max Drawdown Duration                 109 days 00:00:00  << here
Total Trades                                       15.0
Total Closed Trades                                15.0
Total Open Trades                                   0.0
Open Trade PnL                                      0.0
Win Rate [%]                                       67.5
Best Trade [%]                                17.977121
Worst Trade [%]                               -7.432006
Avg Winning Trade [%]                          7.143562
Avg Losing Trade [%]                          -3.400855
Avg Winning Trade Duration    8 days 17:00:39.560439560
Avg Losing Trade Duration               7 days 16:00:00
Profit Factor                                  4.840401
Expectancy                                     4.618165
Sharpe Ratio                                   1.936813
Calmar Ratio                                   8.923935
Omega Ratio                                     1.55757
Sortino Ratio                                  3.440712
Name: agg_func_<lambda>, dtype: object
```

Let's create a simple metric that returns a passed value to demonstrate how vectorbt overrides settings,
from least to most important:

```pycon
>>> # vbt.settings.portfolio.stats
>>> vbt.settings.portfolio.stats['settings']['my_arg'] = 100
>>> my_arg_metric = ('my_arg_metric', dict(title='My Arg', calc_func=lambda my_arg: my_arg))
>>> pf.stats(my_arg_metric, column=10)
My Arg    100
Name: 10, dtype: int64

>>> # settings >>> vbt.settings.portfolio.stats
>>> pf.stats(my_arg_metric, column=10, settings=dict(my_arg=200))
My Arg    200
Name: 10, dtype: int64

>>> # metric settings >>> settings
>>> my_arg_metric = ('my_arg_metric', dict(title='My Arg', my_arg=300, calc_func=lambda my_arg: my_arg))
>>> pf.stats(my_arg_metric, column=10, settings=dict(my_arg=200))
My Arg    300
Name: 10, dtype: int64

>>> # metric_settings >>> metric settings
>>> pf.stats(my_arg_metric, column=10, settings=dict(my_arg=200),
...     metric_settings=dict(my_arg_metric=dict(my_arg=400)))
My Arg    400
Name: 10, dtype: int64
```

Here's an example of a parameterized metric. Let's get the number of trades with PnL over some amount:

```pycon
>>> trade_min_pnl_cnt = (
...     'trade_min_pnl_cnt',
...     dict(
...         title=vbt.Sub('Trades with PnL over $$${min_pnl}'),
...         calc_func=lambda trades, min_pnl: trades.apply_mask(
...             trades.pnl.values >= min_pnl).count(),
...         resolve_trades=True
...     )
... )
>>> pf.stats(
...     metrics=trade_min_pnl_cnt, column=10,
...     metric_settings=dict(trade_min_pnl_cnt=dict(min_pnl=0)))
Trades with PnL over $0    7
Name: 10, dtype: int64

>>> pf.stats(
...     metrics=trade_min_pnl_cnt, column=10,
...     metric_settings=dict(trade_min_pnl_cnt=dict(min_pnl=10)))
Trades with PnL over $10    2
Name: stats, dtype: int64
```

If the same metric name was encountered more than once, vectorbt automatically appends an
underscore and its position, so we can pass keyword arguments to each metric separately:

```pycon
>>> pf.stats(
...     metrics=[
...         trade_min_pnl_cnt,
...         trade_min_pnl_cnt,
...         trade_min_pnl_cnt
...     ],
...     column=10,
...     metric_settings=dict(
...         trade_min_pnl_cnt_0=dict(min_pnl=0),
...         trade_min_pnl_cnt_1=dict(min_pnl=10),
...         trade_min_pnl_cnt_2=dict(min_pnl=20))
...     )
Trades with PnL over $0     7
Trades with PnL over $10    2
Trades with PnL over $20    0
Name: stats, dtype: int64
```

To add a custom metric to the list of all metrics, we have three options.

The first option is to change the `Portfolio.metrics` dict in-place (this will append to the end):

```pycon
>>> pf.metrics['max_winning_streak'] = max_winning_streak[1]
>>> pf.stats(column=10)
Start                         2020-01-01 00:00:00+00:00
End                           2020-08-31 00:00:00+00:00
Period                                243 days 00:00:00
Start Value                                       100.0
End Value                                    139.876426
Total Return [%]                              39.876426
Benchmark Return [%]                          62.229688
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                                12.7421
Max Drawdown Duration                 109 days 00:00:00
Total Trades                                         10
Total Closed Trades                                  10
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                       70.0
Best Trade [%]                                15.303446
Worst Trade [%]                               -9.603504
Avg Winning Trade [%]                          7.372146
Avg Losing Trade [%]                          -4.943456
Avg Winning Trade Duration    7 days 13:42:51.428571428
Avg Losing Trade Duration              12 days 08:00:00
Profit Factor                                  2.941353
Expectancy                                     3.987643
Sharpe Ratio                                   1.515967
Calmar Ratio                                   5.117177
Omega Ratio                                    1.495807
Sortino Ratio                                  2.624107
Max Winning Streak                                  3.0  << here
Name: 10, dtype: object
```

Since `Portfolio.metrics` is of type `vectorbtpro.utils.config.Config`, we can reset it at any time
to get default metrics:

```pycon
>>> pf.metrics.reset()
```

The second option is to copy `Portfolio.metrics`, append our metric, and pass as `metrics` argument:

```pycon
>>> my_metrics = list(pf.metrics.items()) + [max_winning_streak]
>>> pf.stats(metrics=my_metrics, column=10)
```

The third option is to set `metrics` globally under `stats` in `vectorbtpro._settings.portfolio`.

```pycon
>>> vbt.settings.portfolio['stats']['metrics'] = my_metrics
>>> pf.stats(column=10)
```

## Returns stats

We can compute the stats solely based on the portfolio's returns using `Portfolio.returns_stats`,
which calls `vectorbtpro.returns.accessors.ReturnsAccessor.stats`.

```pycon
>>> pf.returns_stats(column=10)
Start                        2020-01-01 00:00:00+00:00
End                          2020-08-31 00:00:00+00:00
Period                               243 days 00:00:00
Total Return [%]                             39.876426
Benchmark Return [%]                         62.229688
Annualized Return [%]                        65.203589
Annualized Volatility [%]                    37.882834
Max Drawdown [%]                               12.7421
Max Drawdown Duration                109 days 00:00:00
Sharpe Ratio                                  1.515967
Calmar Ratio                                  5.117177
Omega Ratio                                   1.495807
Sortino Ratio                                 2.624107
Skew                                          1.817958
Kurtosis                                     14.555089
Tail Ratio                                    1.364631
Common Sense Ratio                             2.25442
Value at Risk                                -0.020681
Alpha                                            0.385
Beta                                          0.233771
Name: 10, dtype: object
```

Most metrics defined in `vectorbtpro.returns.accessors.ReturnsAccessor` are also available
as attributes of `Portfolio`:

```pycon
>>> pf.sharpe_ratio
randnx_n
10    1.515967
20    2.357659
Name: sharpe_ratio, dtype: float64
```

Moreover, we can access quantstats functions using `vectorbtpro.returns.qs_adapter.QSAdapter`:

```pycon
>>> pf.qs.sharpe()
randnx_n
10    1.515967
20    2.357659
dtype: float64
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

    The features implemented in this method are very similar to stats - see [Stats](#stats).

Plot a single column of a portfolio:

```pycon
>>> pf.plot(column=10).show()
```

![](/assets/images/api/portfolio_col_plot.svg)

To plot a single column of a grouped portfolio:

```pycon
>>> pf_grouped = vbt.Portfolio.from_random_signals(
...     close, n=[10, 20, 30, 40], seed=42, freq='d',
...     group_by=['group1', 'group1', 'group2', 'group2'])

>>> pf_grouped.plot(column=10, group_by=False)
```

To plot a single group of a grouped portfolio:

```pycon
>>> pf_grouped.plot(column='group1').show()
UserWarning: Subplot 'orders' does not support grouped data
UserWarning: Subplot 'trade_pnl' does not support grouped data
```

![](/assets/images/api/portfolio_group_plot.svg)

!!! note
    Some subplots do not support plotting grouped data.
    Pass `group_by=False` and select a regular column to plot.

You can choose any of the subplots in `Portfolio.subplots`, in any order, and
control their appearance using keyword arguments:

```pycon
>>> pf.plot(
...     subplots=['drawdowns', 'underwater'],
...     column=10,
...     subplot_settings=dict(
...         drawdowns=dict(top_n=3),
...         underwater=dict(
...             trace_kwargs=dict(
...                 line=dict(color='#FF6F00'),
...                 fillcolor=adjust_opacity('#FF6F00', 0.3)
...             )
...         )
...     )
... ).show()
```

![](/assets/images/api/portfolio_plot_drawdowns.svg)

### Custom subplots

To create a new subplot, a preferred way is to pass a plotting function:

```pycon
>>> def plot_order_size(pf, size, column=None, add_trace_kwargs=None, fig=None):
...     size = pf.select_col_from_obj(size, column, wrapper=pf.wrapper.regroup(False))
...     size.rename('Order Size').vbt.barplot(add_trace_kwargs=add_trace_kwargs, fig=fig)

>>> order_size = pf.orders.size.to_pd(fill_value=0.)
>>> pf.plot(subplots=[
...     'orders',
...     ('order_size', dict(
...         title='Order Size',
...         yaxis_kwargs=dict(title='Order size'),
...         check_is_not_grouped=True,
...         plot_func=plot_order_size
...     ))
... ],
...     column=10,
...     subplot_settings=dict(
...         order_size=dict(
...             size=order_size
...         )
...     )
... )
```

Alternatively, you can create a placeholder and overwrite it manually later:

```pycon
>>> fig = pf.plot(subplots=[
...     'orders',
...     ('order_size', dict(
...         title='Order Size',
...         yaxis_kwargs=dict(title='Order size'),
...         check_is_not_grouped=True
...     ))  # placeholder
... ], column=10)
>>> order_size[10].rename('Order Size').vbt.barplot(
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... ).show()
```

![](/assets/images/api/portfolio_plot_custom.svg)

If a plotting function can in any way be accessed from the current portfolio, you can pass
the path to this function (see `vectorbtpro.utils.attr_.deep_getattr` for the path format).
You can additionally use templates to make some parameters to depend upon passed keyword arguments:

```pycon
>>> subplots = [
...     ('cumulative_returns', dict(
...         title='Cumulative Returns',
...         yaxis_kwargs=dict(title='Cumulative returns'),
...         plot_func='cumulative_returns.vbt.plot',
...         select_col_cumulative_returns=True,
...         pass_add_trace_kwargs=True
...     )),
...     ('rolling_drawdown', dict(
...         title='Rolling Drawdown',
...         yaxis_kwargs=dict(title='Rolling drawdown'),
...         plot_func=[
...             'returns_acc',  # returns accessor
...             (
...                 'rolling_max_drawdown',  # function name
...                 (vbt.Rep('window'),)),  # positional arguments
...             'vbt.plot'  # plotting function
...         ],
...         select_col_returns_acc=True,
...         pass_add_trace_kwargs=True,
...         trace_names=[vbt.Sub('rolling_drawdown(${window})')],  # add window to the trace name
...     ))
... ]
>>> pf.plot(
...     subplots,
...     column=10,
...     subplot_settings=dict(
...         rolling_drawdown=dict(
...             template_context=dict(
...                 window=10
...             )
...         )
...     )
... ).show_svg()
```

You can also replace templates across all subplots by using the global template mapping:

```pycon
>>> pf.plot(subplots, column=10, template_context=dict(window=10)).show()
```

![](/assets/images/api/portfolio_plot_path.svg)
"""

import string
import inspect
import warnings
from collections import namedtuple
from functools import partial
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array, broadcast, broadcast_to, to_pd_array, to_2d_shape, BCO
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.grouping.base import ExceptLevel
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.data.base import Data
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import require_call_seq, build_call_seq
from vectorbtpro.portfolio.decorators import attach_shortcut_properties, attach_returns_acc_methods
from vectorbtpro.portfolio.enums import *
from vectorbtpro.portfolio.logs import Logs
from vectorbtpro.portfolio.orders import Orders, FSOrders
from vectorbtpro.portfolio.trades import Trades, EntryTrades, ExitTrades, Positions
from vectorbtpro.portfolio.pfopt.base import PortfolioOptimizer
from vectorbtpro.records import nb as records_nb
from vectorbtpro.records.base import Records
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.signals.generators import RANDNX, RPROBNX
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig, atomic_dict
from vectorbtpro.utils.datetime_ import freq_to_timedelta64
from vectorbtpro.utils.decorators import custom_property, cached_property, class_or_instancemethod
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.mapping import to_mapping
from vectorbtpro.utils.parsing import get_func_kwargs, get_func_arg_names
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import Rep, RepEval, RepFunc, deep_substitute
from vectorbtpro.utils.chunking import ArgsTaker

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from vectorbtpro.returns.qs_adapter import QSAdapter as QSAdapterT
except ImportError:
    QSAdapterT = tp.Any

__pdoc__ = {}


def fix_wrapper_for_records(pf: "Portfolio") -> ArrayWrapper:
    """Allow flags for records that were restricted for portfolio."""
    if pf.cash_sharing:
        return pf.wrapper.replace(allow_enable=True, allow_modify=True)
    return pf.wrapper


def records_indexing_func(
    self: "Portfolio",
    obj: tp.RecordArray,
    wrapper_meta: dict,
    cls: tp.Union[type, str],
    groups_only: bool = False,
    **kwargs,
) -> tp.RecordArray:
    """Apply indexing function on records."""
    wrapper = fix_wrapper_for_records(self)
    if groups_only:
        wrapper = wrapper.resolve()
        wrapper_meta = dict(wrapper_meta)
        wrapper_meta["col_idxs"] = wrapper_meta["group_idxs"]
    if isinstance(cls, str):
        cls = getattr(self, cls)
    records = cls(wrapper, obj)
    records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
    return records.indexing_func(records_meta=records_meta).values


def records_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    cls: tp.Union[type, str],
    **kwargs,
) -> tp.RecordArray:
    """Apply resampling function on records."""
    if isinstance(cls, str):
        cls = getattr(self, cls)
    return cls(wrapper, obj).resample(resampler).values


def returns_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    fill_with_zero: bool = True,
    **kwargs,
):
    """Apply resampling function on returns."""
    return (
        pd.DataFrame(obj, index=wrapper.index)
        .vbt.returns.resample(
            resampler,
            fill_with_zero=fill_with_zero,
        )
        .obj.values
    )


returns_acc_config = ReadonlyConfig(
    {
        "daily_returns": dict(source_name="daily"),
        "annual_returns": dict(source_name="annual"),
        "cumulative_returns": dict(source_name="cumulative"),
        "annualized_return": dict(source_name="annualized"),
        "annualized_volatility": dict(),
        "calmar_ratio": dict(),
        "omega_ratio": dict(),
        "sharpe_ratio": dict(),
        "sharpe_ratio_std": dict(),
        "prob_sharpe_ratio": dict(),
        "deflated_sharpe_ratio": dict(),
        "downside_risk": dict(),
        "sortino_ratio": dict(),
        "information_ratio": dict(),
        "beta": dict(),
        "alpha": dict(),
        "tail_ratio": dict(),
        "value_at_risk": dict(),
        "cond_value_at_risk": dict(),
        "capture": dict(),
        "up_capture": dict(),
        "down_capture": dict(),
        "drawdown": dict(),
        "max_drawdown": dict(),
    }
)
"""_"""

__pdoc__[
    "returns_acc_config"
] = f"""Config of returns accessor methods to be attached to `Portfolio`.

```python
{returns_acc_config.prettify()}
```
"""

shortcut_config = ReadonlyConfig(
    {
        "filled_close": dict(group_by_aware=False, decorator=cached_property),
        "filled_bm_close": dict(group_by_aware=False, decorator=cached_property),
        "orders": dict(
            obj_type="records",
            field_aliases=("order_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.orders_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="orders_cls"),
            resample_func=partial(records_resample_func, cls="orders_cls"),
        ),
        "logs": dict(
            obj_type="records",
            field_aliases=("log_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.logs_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="logs_cls"),
            resample_func=partial(records_resample_func, cls="logs_cls"),
        ),
        "entry_trades": dict(
            obj_type="records",
            field_aliases=("entry_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.entry_trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="entry_trades_cls"),
            resample_func=partial(records_resample_func, cls="entry_trades_cls"),
        ),
        "exit_trades": dict(
            obj_type="records",
            field_aliases=("exit_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.exit_trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="exit_trades_cls"),
            resample_func=partial(records_resample_func, cls="exit_trades_cls"),
        ),
        "trades": dict(
            obj_type="records",
            field_aliases=("trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="trades_cls"),
            resample_func=partial(records_resample_func, cls="trades_cls"),
        ),
        "trade_history": dict(),
        "positions": dict(
            obj_type="records",
            field_aliases=("position_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.positions_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="positions_cls"),
            resample_func=partial(records_resample_func, cls="positions_cls"),
        ),
        "drawdowns": dict(
            obj_type="records",
            field_aliases=("drawdown_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.drawdowns_cls.from_records(pf.orders.wrapper.resolve(), obj),
            indexing_func=partial(records_indexing_func, cls="drawdowns_cls", groups_only=True),
            resample_func=partial(records_resample_func, cls="drawdowns_cls"),
        ),
        "init_position": dict(obj_type="red_array", group_by_aware=False),
        "asset_flow": dict(
            group_by_aware=False,
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "longonly_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "shortonly_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "assets": dict(group_by_aware=False),
        "longonly_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
        ),
        "shortonly_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
        ),
        "position_mask": dict(),
        "longonly_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="longonly")),
        "shortonly_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="shortonly")),
        "position_coverage": dict(obj_type="red_array"),
        "longonly_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="longonly"),
        ),
        "shortonly_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="shortonly"),
        ),
        "init_cash": dict(obj_type="red_array"),
        "cash_deposits": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "cash_earnings": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "cash_flow": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "free_cash_flow": dict(
            method_name="get_cash_flow",
            method_kwargs=dict(free=True),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "cash": dict(),
        "free_cash": dict(method_name="get_cash", method_kwargs=dict(free=True)),
        "init_price": dict(obj_type="red_array", group_by_aware=False),
        "init_position_value": dict(obj_type="red_array", group_by_aware=False),
        "init_value": dict(obj_type="red_array"),
        "input_value": dict(obj_type="red_array"),
        "asset_value": dict(),
        "longonly_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="longonly")),
        "shortonly_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="shortonly")),
        "gross_exposure": dict(),
        "longonly_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="longonly")),
        "shortonly_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="shortonly")),
        "net_exposure": dict(),
        "value": dict(),
        "allocations": dict(group_by_aware=False),
        "longonly_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="longonly"),
            group_by_aware=False,
        ),
        "shortonly_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="shortonly"),
            group_by_aware=False,
        ),
        "total_profit": dict(obj_type="red_array"),
        "final_value": dict(obj_type="red_array"),
        "total_return": dict(obj_type="red_array"),
        "returns": dict(resample_func=returns_resample_func),
        "asset_pnl": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "asset_returns": dict(resample_func=returns_resample_func),
        "market_value": dict(),
        "market_returns": dict(resample_func=returns_resample_func),
        "bm_value": dict(),
        "bm_returns": dict(resample_func=returns_resample_func),
        "total_market_return": dict(obj_type="red_array"),
        "daily_returns": dict(resample_func=returns_resample_func),
        "annual_returns": dict(resample_func=returns_resample_func),
        "cumulative_returns": dict(),
        "annualized_return": dict(obj_type="red_array"),
        "annualized_volatility": dict(obj_type="red_array"),
        "calmar_ratio": dict(obj_type="red_array"),
        "omega_ratio": dict(obj_type="red_array"),
        "sharpe_ratio": dict(obj_type="red_array"),
        "sharpe_ratio_std": dict(obj_type="red_array"),
        "prob_sharpe_ratio": dict(obj_type="red_array"),
        "deflated_sharpe_ratio": dict(obj_type="red_array"),
        "downside_risk": dict(obj_type="red_array"),
        "sortino_ratio": dict(obj_type="red_array"),
        "information_ratio": dict(obj_type="red_array"),
        "beta": dict(obj_type="red_array"),
        "alpha": dict(obj_type="red_array"),
        "tail_ratio": dict(obj_type="red_array"),
        "value_at_risk": dict(obj_type="red_array"),
        "cond_value_at_risk": dict(obj_type="red_array"),
        "capture": dict(obj_type="red_array"),
        "up_capture": dict(obj_type="red_array"),
        "down_capture": dict(obj_type="red_array"),
        "drawdown": dict(),
        "max_drawdown": dict(obj_type="red_array"),
    }
)
"""_"""

__pdoc__[
    "shortcut_config"
] = f"""Config of shortcut properties to be attached to `Portfolio`.

```python
{shortcut_config.prettify()}
```
"""

PortfolioT = tp.TypeVar("PortfolioT", bound="Portfolio")


class MetaInOutputs(type):
    """Meta class that exposes a read-only class property `MetaFields.in_output_config`."""

    @property
    def in_output_config(cls) -> Config:
        """In-output config."""
        return cls._in_output_config


class PortfolioWithInOutputs(metaclass=MetaInOutputs):
    """Class exposes a read-only class property `RecordsWithFields.field_config`."""

    @property
    def in_output_config(self) -> Config:
        """In-output config of `${cls_name}`.

        ```python
        ${in_output_config}
        ```
        """
        return self._in_output_config


class MetaPortfolio(type(Analyzable), type(PortfolioWithInOutputs)):
    pass


@attach_shortcut_properties(shortcut_config)
@attach_returns_acc_methods(returns_acc_config)
class Portfolio(Analyzable, PortfolioWithInOutputs, metaclass=MetaPortfolio):
    """Class for modeling portfolio and measuring its performance.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        close (array_like): Last asset price at each time step.
        order_records (array_like): A structured NumPy array of order records.
        open (array_like): Open price of each bar.
        high (array_like): High price of each bar.
        low (array_like): Low price of each bar.
        log_records (array_like): A structured NumPy array of log records.
        cash_sharing (bool): Whether to share cash within the same group.
        init_cash (InitCashMode or array_like of float): Initial capital.

            Can be provided in a format suitable for flexible indexing.
        init_position (array_like of float): Initial position.

            Can be provided in a format suitable for flexible indexing.
        init_price (array_like of float): Initial position price.

            Can be provided in a format suitable for flexible indexing.
        cash_deposits (array_like of float): Cash deposited/withdrawn at each timestamp.

            Can be provided in a format suitable for flexible indexing.
        cash_earnings (array_like of float): Earnings added at each timestamp.

            Can be provided in a format suitable for flexible indexing.
        call_seq (array_like of int): Sequence of calls per row and group. Defaults to None.
        in_outputs (namedtuple): Named tuple with in-output objects.

            To substitute `Portfolio` attributes, provide already broadcasted and grouped objects.
            Also see `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.
        use_in_outputs (bool): Whether to return in-output objects when calling properties.
        bm_close (array_like): Last benchmark asset price at each time step.
        fillna_close (bool): Whether to forward and backward fill NaN values in `close`.

            Applied after the simulation to avoid NaNs in asset value.

            See `Portfolio.get_filled_close`.
        trades_type (str or int): Default `vectorbtpro.portfolio.trades.Trades` to use across `Portfolio`.

            See `vectorbtpro.portfolio.enums.TradesType`.
        orders_cls (type): Class for wrapping order records.
        logs_cls (type): Class for wrapping log records.
        trades_cls (type): Class for wrapping trade records.
        entry_trades_cls (type): Class for wrapping entry trade records.
        exit_trades_cls (type): Class for wrapping exit trade records.
        positions_cls (type): Class for wrapping position records.
        drawdowns_cls (type): Class for wrapping drawdown records.

    For defaults, see `vectorbtpro._settings.portfolio`.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Portfolio.replace`."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_in_output_config"}

    @classmethod
    def row_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        row_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack (two-dimensional) objects along rows.

        `row_stack_func` must take the portfolio class, and all the arguments passed to this method.
        If you don't need any of the arguments, make `row_stack_func` accept them as `**kwargs`.

        If all the objects are None, boolean, or empty, returns the first one."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if row_stack_func is not None:
            return row_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(objs[0]):
            n_cols = wrapper.get_shape_2d(group_by=obj_group_by)[1]
            can_stack = (objs[0].ndim == 1 and n_cols == 1) or (objs[0].ndim == 2 and objs[0].shape[1] == n_cols)
        elif obj_type is not None and obj_type == "array":
            can_stack = True
        else:
            can_stack = False
        if can_stack:
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.row_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along rows")

    @classmethod
    def row_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` along rows.

        All in-output tuples must be either None or have the same fields.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs stacking on the in-output objects of the same field using `Portfolio.row_stack_objs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                row_stack_func = prop_options.get("row_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                row_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    row_stack_func=field_options.get("row_stack_func", row_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.row_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @classmethod
    def row_stack(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        combine_init_cash: bool = False,
        combine_init_position: bool = False,
        combine_init_price: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        Cash sharing must be the same among all objects.

        Close, benchmark close, cash deposits, cash earnings, call sequence, and other two-dimensional arrays
        are stacked using `vectorbtpro.base.wrapping.ArrayWrapper.row_stack_arrs`. In-outputs
        are stacked using `Portfolio.row_stack_in_outputs`. Records are stacked using
        `vectorbtpro.records.base.Records.row_stack_records_arrs`.

        If the initial cash of each object is one of the options in `vectorbtpro.portfolio.enums.InitCashMode`,
        it will be retained for the resulting object. Once any of the objects has the initial cash listed
        as an absolute amount or an array, the initial cash of the first object will be copied over to
        the final object, while the initial cash of all other objects will be resolved and used
        as cash deposits, unless they all are zero. Set `combine_init_cash` to True to simply sum all
        initial cash arrays.

        If only the first object has an initial position greater than zero, it will be copied over to
        the final object. Otherwise, an error will be thrown, unless `combine_init_position` is enabled
        to sum all initial position arrays. The same goes for the initial price, which becomes
        a candidate for stacking only if any of the arrays are not NaN.

        !!! note
            When possible, avoid using initial position and price in objects to be stacked:
            there is currently no way of injecting them in the correct order, while simply taking
            the sum or weighted average may distort the reality since they weren't available
            prior to the actual simulation."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        for i in range(1, len(objs)):
            if objs[i].cash_sharing != objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        kwargs["cash_sharing"] = objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False
        cs_n_cols = kwargs["wrapper"].get_shape_2d(group_by=cs_group_by)[1]
        n_cols = kwargs["wrapper"].shape_2d[1]

        if "close" not in kwargs:
            kwargs["close"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.close for obj in objs],
                group_by=False,
                wrap=False,
            )
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.open for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.high for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.low for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.row_stack_records_arrs(*[obj.orders for obj in objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.row_stack_records_arrs(*[obj.logs for obj in objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                stack_init_cash_objs = False
                init_cash_objs = []
                for i, obj in enumerate(objs):
                    init_cash_obj = obj.get_init_cash(group_by=cs_group_by)
                    init_cash_obj = to_1d_array(init_cash_obj)
                    init_cash_obj = np.broadcast_to(init_cash_obj, cs_n_cols)
                    if i > 0 and (init_cash_obj != 0).any():
                        stack_init_cash_objs = True
                    init_cash_objs.append(init_cash_obj)
                if stack_init_cash_objs:
                    if not combine_init_cash:
                        cash_deposits_objs = []
                        for i, obj in enumerate(objs):
                            cash_deposits_obj = obj.get_cash_deposits(group_by=cs_group_by)
                            cash_deposits_obj = to_2d_array(cash_deposits_obj)
                            cash_deposits_obj = np.broadcast_to(
                                cash_deposits_obj,
                                (cash_deposits_obj.shape[0], cs_n_cols),
                            )
                            cash_deposits_obj = cash_deposits_obj.copy()
                            if i > 0:
                                cash_deposits_obj[0] = init_cash_objs[i]
                            cash_deposits_objs.append(cash_deposits_obj)
                        kwargs["cash_deposits"] = np.row_stack(cash_deposits_objs)
                        kwargs["init_cash"] = init_cash_objs[0]
                    else:
                        kwargs["init_cash"] = np.asarray(init_cash_objs).sum(axis=0)
                else:
                    kwargs["init_cash"] = init_cash_objs[0]
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            init_position_objs = []
            for i, obj in enumerate(objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = np.broadcast_to(init_position_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any():
                    stack_init_position_objs = True
                init_position_objs.append(init_position_obj)
            if stack_init_position_objs:
                if not combine_init_position:
                    raise ValueError("Initial position cannot be stacked along rows")
                kwargs["init_position"] = np.asarray(init_position_objs).sum(axis=0)
            else:
                kwargs["init_position"] = init_position_objs[0]
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            init_position_objs = []
            init_price_objs = []
            for i, obj in enumerate(objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = np.broadcast_to(init_position_obj, n_cols)
                init_price_obj = obj.get_init_price()
                init_price_obj = to_1d_array(init_price_obj)
                init_price_obj = np.broadcast_to(init_price_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any() and not np.isnan(init_price_obj).all():
                    stack_init_price_objs = True
                init_position_objs.append(init_position_obj)
                init_price_objs.append(init_price_obj)
            if stack_init_price_objs:
                if not combine_init_price:
                    raise ValueError("Initial price cannot be stacked along rows")
                init_position_objs = np.asarray(init_position_objs)
                init_price_objs = np.asarray(init_price_objs)
                mask1 = (init_position_objs != 0).any(axis=1)
                mask2 = (~np.isnan(init_price_objs)).any(axis=1)
                mask = mask1 & mask2
                init_position_objs = init_position_objs[mask]
                init_price_objs = init_price_objs[mask]
                nom = (init_position_objs * init_price_objs).sum(axis=0)
                denum = init_position_objs.sum(axis=0)
                kwargs["init_price"] = nom / denum
            else:
                kwargs["init_price"] = init_price_objs[0]
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in objs],
                    group_by=cs_group_by,
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in objs],
                    group_by=False,
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.call_seq for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                kwargs["bm_close"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.bm_close for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.row_stack_in_outputs(*objs, **kwargs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        column_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack (one and two-dimensional) objects along column.

        `column_stack_func` must take the portfolio class, and all the arguments passed to this method.
        If you don't need any of the arguments, make `column_stack_func` accept them as `**kwargs`.

        If all the objects are None, boolean, or empty, returns the first one."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if column_stack_func is not None:
            return column_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(obj):
            n_cols = wrapper.get_shape_2d(group_by=obj_group_by)[1]
            if to_2d_shape(objs[0].shape) == wrappers[0].get_shape_2d(group_by=obj_group_by):
                can_stack = True
                reduced = False
            elif objs[0].shape == (wrappers[0].get_shape_2d(group_by=obj_group_by)[1],):
                can_stack = True
                reduced = True
            else:
                can_stack = False
        elif obj_type is not None and obj_type == "array":
            can_stack = True
            reduced = False
        elif obj_type is not None and obj_type == "red_array":
            can_stack = True
            reduced = True
        else:
            can_stack = False
        if can_stack:
            if reduced:
                wrapped_objs = []
                for i, obj in enumerate(objs):
                    wrapped_objs.append(wrappers[i].wrap_reduced(obj, group_by=obj_group_by))
                return wrapper.concat_arrs(*wrapped_objs, group_by=obj_group_by).values
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.column_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along columns")

    @classmethod
    def column_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` along columns.

        All in-output tuples must be either None or have the same fields.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs stacking on the in-output objects of the same field using `Portfolio.column_stack_objs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                column_stack_func = prop_options.get("column_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                column_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    column_stack_func=field_options.get("column_stack_func", column_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.column_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @classmethod
    def column_stack(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        Cash sharing must be the same among all objects.

        Two-dimensional arrays are stacked using
        `vectorbtpro.base.wrapping.ArrayWrapper.column_stack_arrs`
        while one-dimensional arrays are stacked using
        `vectorbtpro.base.wrapping.ArrayWrapper.concat_arrs`.
        In-outputs are stacked using `Portfolio.column_stack_in_outputs`. Records are stacked using
        `vectorbtpro.records.base.Records.column_stack_records_arrs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        for i in range(1, len(objs)):
            if objs[i].cash_sharing != objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        kwargs["cash_sharing"] = objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False

        if "close" not in kwargs:
            new_close = kwargs["wrapper"].column_stack_arrs(
                *[obj.close for obj in objs],
                group_by=False,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
            kwargs["close"] = new_close
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.open for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.high for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.low for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.column_stack_records_arrs(*[obj.orders for obj in objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.column_stack_records_arrs(*[obj.logs for obj in objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                kwargs["init_cash"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.get_init_cash(group_by=cs_group_by) for obj in objs],
                        group_by=cs_group_by,
                    )
                )
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            for obj in objs:
                if (to_1d_array(obj.init_position) != 0).any():
                    stack_init_position_objs = True
                    break
            if stack_init_position_objs:
                kwargs["init_position"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_position for obj in objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_position"] = np.array([0.0])
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            for obj in objs:
                if not np.isnan(to_1d_array(obj.init_price)).all():
                    stack_init_price_objs = True
                    break
            if stack_init_price_objs:
                kwargs["init_price"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_price for obj in objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_price"] = np.array([np.nan])
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in objs],
                    group_by=cs_group_by,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.call_seq for obj in objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                new_bm_close = kwargs["wrapper"].column_stack_arrs(
                    *[obj.bm_close for obj in objs],
                    group_by=False,
                    wrap=False,
                )
                if fbfill_close:
                    new_bm_close = new_bm_close.vbt.fbfill()
                elif ffill_close:
                    new_bm_close = new_bm_close.vbt.ffill()
                kwargs["bm_close"] = new_bm_close
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.column_stack_in_outputs(*objs, **kwargs)

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "close",
        "order_records",
        "open",
        "high",
        "low",
        "log_records",
        "cash_sharing",
        "init_cash",
        "init_position",
        "init_price",
        "cash_deposits",
        "cash_earnings",
        "call_seq",
        "in_outputs",
        "use_in_outputs",
        "bm_close",
        "fillna_close",
        "trades_type",
        "orders_cls",
        "logs_cls",
        "trades_cls",
        "entry_trades_cls",
        "exit_trades_cls",
        "positions_cls",
        "drawdowns_cls",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        close: tp.ArrayLike,
        order_records: tp.RecordArray,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        log_records: tp.Optional[tp.RecordArray] = None,
        cash_sharing: bool = False,
        init_cash: tp.Union[str, tp.ArrayLike] = "auto",
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        cash_deposits: tp.ArrayLike = 0.0,
        cash_earnings: tp.ArrayLike = 0.0,
        call_seq: tp.Optional[tp.Array2d] = None,
        in_outputs: tp.Optional[tp.NamedTuple] = None,
        use_in_outputs: tp.Optional[bool] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        fillna_close: tp.Optional[bool] = None,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        orders_cls: type = Orders,
        logs_cls: type = Logs,
        trades_cls: type = Trades,
        entry_trades_cls: type = EntryTrades,
        exit_trades_cls: type = ExitTrades,
        positions_cls: type = Positions,
        drawdowns_cls: type = Drawdowns,
        **kwargs,
    ) -> None:

        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if cash_sharing:
            if wrapper.grouper.allow_enable or wrapper.grouper.allow_modify:
                wrapper = wrapper.replace(allow_enable=False, allow_modify=False)
        Analyzable.__init__(
            self,
            wrapper,
            close=close,
            order_records=order_records,
            open=open,
            high=high,
            low=low,
            log_records=log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            call_seq=call_seq,
            in_outputs=in_outputs,
            use_in_outputs=use_in_outputs,
            bm_close=bm_close,
            fillna_close=fillna_close,
            trades_type=trades_type,
            orders_cls=orders_cls,
            logs_cls=logs_cls,
            trades_cls=trades_cls,
            entry_trades_cls=entry_trades_cls,
            exit_trades_cls=exit_trades_cls,
            positions_cls=positions_cls,
            drawdowns_cls=drawdowns_cls,
            **kwargs,
        )

        close = to_2d_array(close)
        if open is not None:
            open = to_2d_array(open)
        if high is not None:
            high = to_2d_array(high)
        if low is not None:
            low = to_2d_array(low)
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if not checks.is_int(init_cash) or init_cash not in InitCashMode:
            init_cash = to_1d_array(init_cash)
        init_position = to_1d_array(init_position)
        init_price = to_1d_array(init_price)
        cash_deposits = to_2d_array(cash_deposits)
        cash_earnings = to_2d_array(cash_earnings)
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = to_2d_array(bm_close)
        if log_records is None:
            log_records = np.array([], dtype=log_dt)
        if use_in_outputs is None:
            use_in_outputs = portfolio_cfg["use_in_outputs"]
        if fillna_close is None:
            fillna_close = portfolio_cfg["fillna_close"]
        if trades_type is None:
            trades_type = portfolio_cfg["trades_type"]
        if isinstance(trades_type, str):
            trades_type = map_enum_fields(trades_type, TradesType)

        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._order_records = order_records
        self._log_records = log_records
        self._cash_sharing = cash_sharing
        self._init_cash = init_cash
        self._init_position = init_position
        self._init_price = init_price
        self._cash_deposits = cash_deposits
        self._cash_earnings = cash_earnings
        self._call_seq = call_seq
        self._in_outputs = in_outputs
        self._use_in_outputs = use_in_outputs
        self._bm_close = bm_close
        self._fillna_close = fillna_close
        self._trades_type = trades_type
        self._orders_cls = orders_cls
        self._logs_cls = logs_cls
        self._trades_cls = trades_cls
        self._entry_trades_cls = entry_trades_cls
        self._exit_trades_cls = exit_trades_cls
        self._positions_cls = positions_cls
        self._drawdowns_cls = drawdowns_cls

        # Only slices of rows can be selected
        self._range_only_select = True

        # Copy writeable attrs
        self._in_output_config = type(self)._in_output_config.copy()

    # ############# In-outputs ############# #

    _in_output_config: tp.ClassVar[Config] = HybridConfig(
        dict(
            returns=dict(
                grouping="cash_sharing",
            ),
        )
    )

    @property
    def in_output_config(self) -> Config:
        """In-output config of `${cls_name}`.

        ```python
        ${in_output_config}
        ```

        Returns `${cls_name}._in_output_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change in_outputs, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._in_output_config`.
        """
        return self._in_output_config

    @classmethod
    def parse_field_options(cls, field: str) -> tp.Kwargs:
        """Parse options based on the name of a field.

        Returns a dictionary with the parsed grouping, object type, and cleaned field name.

        Grouping is parsed by looking for the following suffixes:

        * '_cs': per group if grouped with cash sharing, otherwise per column
        * '_pcg': per group if grouped, otherwise per column
        * '_pg': per group
        * '_pc': per column
        * '_records': records

        Object type is parsed by looking for the following suffixes:

        * '_2d': element per timestamp and column/group (time series)
        * '_1d': element per column/group (reduced time series)

        Those substrings are then removed to produce a clean field name."""
        options = dict()
        new_parts = []
        for part in field.split("_"):
            if part == "1d":
                options["obj_type"] = "red_array"
            elif part == "2d":
                options["obj_type"] = "array"
            elif part == "records":
                options["obj_type"] = "records"
            elif part == "pc":
                options["grouping"] = "columns"
            elif part == "pg":
                options["grouping"] = "groups"
            elif part == "pcg":
                options["grouping"] = "columns_or_groups"
            elif part == "cs":
                options["grouping"] = "cash_sharing"
            else:
                new_parts.append(part)
        field = "_".join(new_parts)
        options["field"] = field
        return options

    def matches_field_options(
        self,
        options: tp.Kwargs,
        obj_type: tp.Optional[str] = None,
        group_by_aware: bool = True,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> bool:
        """Return whether options of a field match the requirements.

        Requirements include the type of the object (array, reduced array, records),
        the grouping of the object (1/2 dimensions, group/column-wise layout). The current
        grouping and cash sharing of this portfolio object are also taken into account.

        When an option is not in `options`, it's automatically marked as matching."""
        field_obj_type = options.get("obj_type", None)
        field_grouping = options.get("grouping", None)
        if field_obj_type is not None and obj_type is not None:
            if field_obj_type != obj_type:
                return False
        if field_grouping is not None:
            if wrapper is None:
                wrapper = self.wrapper
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if is_grouped:
                if group_by_aware:
                    if field_grouping == "groups":
                        return True
                    if field_grouping == "columns_or_groups":
                        return True
                    if self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
                else:
                    if field_grouping == "columns":
                        return True
                    if not self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
            else:
                if field_grouping == "columns":
                    return True
                if field_grouping == "columns_or_groups":
                    return True
                if field_grouping == "cash_sharing":
                    return True
            return False
        return True

    def wrap_obj(
        self,
        obj: tp.Any,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_func: tp.Optional[tp.Callable] = None,
        wrap_kwargs: tp.KwargsLike = None,
        force_wrapping: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Wrap an object.

        `wrap_func` must take the portfolio, `obj`, all the arguments passed to this method, and `**kwargs`.
        If you don't need any of the arguments, make `indexing_func` accept them as `**kwargs`.

        If the object is None or boolean, returns as-is."""
        if obj is None or isinstance(obj, bool):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if wrap_func is not None:
            return wrap_func(
                self,
                obj,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
                force_wrapping=force_wrapping,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _wrap_1d_grouped(obj: tp.Array) -> tp.Series:
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=group_by, **_wrap_kwargs)

        def _wrap_1d(obj: tp.Array) -> tp.Series:
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=False, **_wrap_kwargs)

        def _wrap_2d_grouped(obj: tp.Array) -> tp.Frame:
            return wrapper.wrap(obj, group_by=group_by, **resolve_dict(wrap_kwargs))

        def _wrap_2d(obj: tp.Array) -> tp.Frame:
            return wrapper.wrap(obj, group_by=False, **resolve_dict(wrap_kwargs))

        if obj_type is not None and obj_type not in {"records"}:
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if grouping == "cash_sharing":
                if obj_type == "array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_2d_grouped(obj)
                    return _wrap_2d(obj)
                if obj_type == "red_array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_1d_grouped(obj)
                    return _wrap_1d(obj)
                if obj.ndim == 2:
                    if is_grouped and self.cash_sharing:
                        return _wrap_2d_grouped(obj)
                    return _wrap_2d(obj)
                if obj.ndim == 1:
                    if is_grouped and self.cash_sharing:
                        return _wrap_1d_grouped(obj)
                    return _wrap_1d(obj)
            if grouping == "columns_or_groups":
                if obj_type == "array":
                    if is_grouped:
                        return _wrap_2d_grouped(obj)
                    return _wrap_2d(obj)
                if obj_type == "red_array":
                    if is_grouped:
                        return _wrap_1d_grouped(obj)
                    return _wrap_1d(obj)
                if obj.ndim == 2:
                    if is_grouped:
                        return _wrap_2d_grouped(obj)
                    return _wrap_2d(obj)
                if obj.ndim == 1:
                    if is_grouped:
                        return _wrap_1d_grouped(obj)
                    return _wrap_1d(obj)
            if grouping == "groups":
                if obj_type == "array":
                    return _wrap_2d_grouped(obj)
                if obj_type == "red_array":
                    return _wrap_1d_grouped(obj)
                if obj.ndim == 2:
                    return _wrap_2d_grouped(obj)
                if obj.ndim == 1:
                    return _wrap_1d_grouped(obj)
            if grouping == "columns":
                if obj_type == "array":
                    return _wrap_2d(obj)
                if obj_type == "red_array":
                    return _wrap_1d(obj)
                if obj.ndim == 2:
                    return _wrap_2d(obj)
                if obj.ndim == 1:
                    return _wrap_1d(obj)
            if checks.is_np_array(obj):
                if is_grouped:
                    if obj_type == "array":
                        return _wrap_2d_grouped(obj)
                    if obj_type == "red_array":
                        return _wrap_1d_grouped(obj)
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _wrap_2d_grouped(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return _wrap_1d_grouped(obj)
                if obj_type == "array":
                    return _wrap_2d(obj)
                if obj_type == "red_array":
                    return _wrap_1d(obj)
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _wrap_2d(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return _wrap_1d(obj)
        if force_wrapping:
            raise NotImplementedError(f"Cannot wrap object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to wrap object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def get_in_output(
        self,
        field: str,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> tp.Union[None, bool, tp.AnyArray]:
        """Find and wrap an in-output object matching the field.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        If `field` is not in `Portfolio.in_outputs`, searches for the field in aliases and options.
        In such case, to narrow down the number of candidates, options are additionally matched against
        the requirements using `Portfolio.matches_field_options`. Finally, the matched in-output object is
        wrapped using `Portfolio.wrap_obj`."""
        if self.in_outputs is None:
            raise ValueError("No in-outputs attached")

        if field in self.cls_dir:
            prop = getattr(type(self), field)
            prop_options = getattr(prop, "options", {})
            obj_type = prop_options.get("obj_type", "array")
            group_by_aware = prop_options.get("group_by_aware", True)
            wrap_func = prop_options.get("wrap_func", None)
            wrap_kwargs = prop_options.get("wrap_kwargs", None)
            force_wrapping = prop_options.get("force_wrapping", False)
            silence_warnings = prop_options.get("silence_warnings", False)
            field_aliases = prop_options.get("field_aliases", None)
            if field_aliases is None:
                field_aliases = []
            field_aliases = {field, *field_aliases}
            found_attr = True
        else:
            obj_type = None
            group_by_aware = True
            wrap_func = None
            wrap_kwargs = None
            force_wrapping = False
            silence_warnings = False
            field_aliases = {field}
            found_attr = False

        found_field = None
        found_field_options = None
        for _field in set(self.in_outputs._fields):
            _field_options = merge_dicts(
                self.parse_field_options(_field),
                self.in_output_config.get(_field, None),
            )
            if (not found_attr and field == _field) or (
                (_field in field_aliases or _field_options.get("field", _field) in field_aliases)
                and self.matches_field_options(
                    _field_options,
                    obj_type=obj_type,
                    group_by_aware=group_by_aware,
                    group_by=group_by,
                    wrapper=wrapper,
                )
            ):
                if found_field is not None:
                    raise ValueError(f"Multiple fields for '{field}' found in in_outputs")
                found_field = _field
                found_field_options = _field_options
        if found_field is None:
            raise AttributeError(f"No compatible field for '{field}' found in in_outputs")
        obj = getattr(self.in_outputs, found_field)
        if found_attr and checks.is_np_array(obj) and obj.shape == (0, 0):  # for returns
            return None
        kwargs = merge_dicts(
            dict(
                grouping=found_field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                obj_type=found_field_options.get("obj_type", obj_type),
                wrap_func=found_field_options.get("wrap_func", wrap_func),
                wrap_kwargs=found_field_options.get("wrap_kwargs", wrap_kwargs),
                force_wrapping=found_field_options.get("force_wrapping", force_wrapping),
                silence_warnings=found_field_options.get("silence_warnings", silence_warnings),
            ),
            kwargs,
        )
        return self.wrap_obj(
            obj,
            found_field_options.get("field", found_field),
            group_by=group_by,
            wrapper=wrapper,
            **kwargs,
        )

    # ############# Indexing ############# #

    def index_obj(
        self,
        obj: tp.Any,
        wrapper_meta: dict,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        indexing_func: tp.Optional[tp.Callable] = None,
        force_indexing: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Perform indexing on an object.

        `indexing_func` must take the portfolio, all the arguments passed to this method, and `**kwargs`.
        If you don't need any of the arguments, make `indexing_func` accept them as `**kwargs`.

        If the object is None, boolean, or empty, returns as-is."""
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if indexing_func is not None:
            return indexing_func(
                self,
                obj,
                wrapper_meta,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                force_indexing=force_indexing,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _index_1d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["group_idxs"]]

        def _index_1d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["col_idxs"]]

        def _index_2d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["group_idxs"]]

        def _index_2d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["col_idxs"]]

        def _index_records(obj: tp.RecordArray) -> tp.RecordArray:
            records = Records(wrapper, obj)
            records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
            return records.indexing_func(records_meta=records_meta).values

        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
        if obj_type is not None and obj_type == "records":
            return _index_records(obj)
        if grouping == "cash_sharing":
            if obj_type is not None and obj_type == "array":
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "columns_or_groups":
            if obj_type is not None and obj_type == "array":
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "groups":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_group(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_group(obj)
            if obj.ndim == 2:
                return _index_2d_by_group(obj)
            if obj.ndim == 1:
                return _index_1d_by_group(obj)
        if grouping == "columns":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                return _index_1d_by_col(obj)
        if checks.is_np_array(obj):
            if is_grouped:
                if obj_type is not None and obj_type == "array":
                    return _index_2d_by_group(obj)
                if obj_type is not None and obj_type == "red_array":
                    return _index_1d_by_group(obj)
                if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                    return _index_2d_by_group(obj)
                if obj.shape == (wrapper.get_shape_2d()[1],):
                    return _index_1d_by_group(obj)
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if to_2d_shape(obj.shape) == wrapper.shape_2d:
                return _index_2d_by_col(obj)
            if obj.shape == (wrapper.shape_2d[1],):
                return _index_1d_by_col(obj)
        if force_indexing:
            raise NotImplementedError(f"Cannot index object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to index object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def in_outputs_indexing_func(self, wrapper_meta: dict, **kwargs) -> tp.Optional[tp.NamedTuple]:
        """Perform indexing on `Portfolio.in_outputs`.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs indexing on the in-output object using `Portfolio.index_obj`."""
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                indexing_func = prop_options.get("indexing_func", None)
                force_indexing = prop_options.get("force_indexing", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                group_by_aware = True
                indexing_func = None
                force_indexing = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    indexing_func=field_options.get("indexing_func", indexing_func),
                    force_indexing=field_options.get("force_indexing", force_indexing),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.index_obj(obj, wrapper_meta, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def indexing_func(
        self: PortfolioT,
        *args,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Perform indexing on `Portfolio`.

        In-outputs are indexed using `Portfolio.in_outputs_indexing_func`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                range_only_select=self.range_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        row_idxs = wrapper_meta["row_idxs"]
        rows_changed = wrapper_meta["rows_changed"]
        col_idxs = wrapper_meta["col_idxs"]
        columns_changed = wrapper_meta["columns_changed"]
        group_idxs = wrapper_meta["group_idxs"]
        groups_changed = wrapper_meta["groups_changed"]

        new_close = ArrayWrapper.select_from_flex_array(
            self._close,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if self._open is not None:
            new_open = ArrayWrapper.select_from_flex_array(
                self._open,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_open = self._open
        if self._high is not None:
            new_high = ArrayWrapper.select_from_flex_array(
                self._high,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_high = self._high
        if self._low is not None:
            new_low = ArrayWrapper.select_from_flex_array(
                self._low,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_low = self._low
        new_order_records = self.orders.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_log_records = self.logs.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_init_cash = self._init_cash
        if not checks.is_int(new_init_cash):
            new_init_cash = to_1d_array(new_init_cash)
            if rows_changed and row_idxs.start > 0:
                if self.wrapper.grouper.is_grouped() and not self.cash_sharing:
                    cash = self.get_cash(group_by=False)
                else:
                    cash = self.cash
                new_init_cash = to_1d_array(cash.iloc[row_idxs.start - 1])
            if columns_changed and new_init_cash.shape[0] > 1:
                if self.cash_sharing:
                    new_init_cash = new_init_cash[group_idxs]
                else:
                    new_init_cash = new_init_cash[col_idxs]
        new_init_position = to_1d_array(self._init_position)
        if rows_changed and row_idxs.start > 0:
            new_init_position = to_1d_array(self.assets.iloc[row_idxs.start - 1])
        if columns_changed and new_init_position.shape[0] > 1:
            new_init_position = new_init_position[col_idxs]
        new_init_price = to_1d_array(self._init_price)
        if rows_changed and row_idxs.start > 0:
            new_init_price = to_1d_array(self.close.iloc[: row_idxs.start].ffill().iloc[-1])
        if columns_changed and new_init_price.shape[0] > 1:
            new_init_price = new_init_price[col_idxs]
        new_cash_deposits = ArrayWrapper.select_from_flex_array(
            self._cash_deposits,
            row_idxs=row_idxs,
            col_idxs=group_idxs if self.cash_sharing else col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        new_cash_earnings = ArrayWrapper.select_from_flex_array(
            self._cash_earnings,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if self._call_seq is not None:
            new_call_seq = ArrayWrapper.select_from_flex_array(
                self._call_seq,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_call_seq = None
        if self._bm_close is not None and not isinstance(self._bm_close, bool):
            new_bm_close = ArrayWrapper.select_from_flex_array(
                self._bm_close,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_bm_close = self._bm_close
        new_in_outputs = self.in_outputs_indexing_func(wrapper_meta, **resolve_dict(in_output_kwargs))

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            close=new_close,
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            log_records=new_log_records,
            init_cash=new_init_cash,
            init_position=new_init_position,
            init_price=new_init_price,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            call_seq=new_call_seq,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
        )

    # ############# Resampling ############# #

    def resample_obj(
        self,
        obj: tp.Any,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        resample_func: tp.Union[None, str, tp.Callable] = None,
        resample_kwargs: tp.KwargsLike = None,
        force_resampling: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Resample an object.

        `resample_func` must take the portfolio, `obj`, `resampler`, all the arguments passed to this method,
        and `**kwargs`. If you don't need any of the arguments, make `resample_func` accept them as `**kwargs`.
        If `resample_func` is a string, will use it as `reduce_func_nb` in
        `vectorbtpro.generic.accessors.GenericAccessor.resample_apply`. Default is 'last'.

        If the object is None, boolean, or empty, returns as-is."""
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if resample_func is None:
            resample_func = "last"
        if not isinstance(resample_func, str):
            return resample_func(
                self,
                obj,
                resampler,
                obj_name=obj_name,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                resample_kwargs=resample_kwargs,
                force_resampling=force_resampling,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _resample(obj: tp.Array) -> tp.SeriesFrame:
            wrapped_obj = ArrayWrapper.from_obj(obj, index=wrapper.index).wrap(obj)
            return wrapped_obj.vbt.resample_apply(resampler, resample_func, **resolve_dict(resample_kwargs)).values

        if obj_type is not None and obj_type == "red_array":
            return obj
        if obj_type is None or obj_type == "array":
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if checks.is_np_array(obj):
                if is_grouped:
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _resample(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return obj
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _resample(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return obj
        if force_resampling:
            raise NotImplementedError(f"Cannot resample object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to resample object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def resample_in_outputs(
        self,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Resample `Portfolio.in_outputs`.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs indexing on the in-output object using `Portfolio.resample_obj`."""
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                resample_func = prop_options.get("resample_func", None)
                resample_kwargs = prop_options.get("resample_kwargs", None)
                force_resampling = prop_options.get("force_resampling", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                resample_func = None
                resample_kwargs = None
                force_resampling = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    resample_func=field_options.get("resample_func", resample_func),
                    resample_kwargs=field_options.get("resample_kwargs", resample_kwargs),
                    force_resampling=field_options.get("force_resampling", force_resampling),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.resample_obj(obj, resampler, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def resample(
        self: PortfolioT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Resample the `Portfolio` instance.

        !!! warning
            Downsampling is associated with information loss:

            * Cash deposits and earnings are assumed to be added/removed at the beginning of each time step.
                Imagine depositing $100 and using them up in the same bar, and then depositing another $100
                and using them up. Downsampling both bars into a single bar will aggregate cash deposits
                and earnings, and put both of them at the beginning of the new bar, even though the second
                deposit was added later in time.
            * Market/benchmark returns are computed by applying the initial value on the close price
                of the first bar and by tracking the price change to simulate holding. Moving the close
                price of the first bar further into the future will affect this computation and almost
                certainly produce a different market value and returns. To mitigate this, make sure
                to downsample to an index with the first bar containing only the first bar from the
                origin timeframe."""
        if self._call_seq is not None:
            raise ValueError("Cannot resample call_seq")
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        resampler = wrapper_meta["resampler"]
        new_wrapper = wrapper_meta["new_wrapper"]

        new_close = self.close.vbt.resample_apply(resampler, "last")
        if fbfill_close:
            new_close = new_close.vbt.fbfill()
        elif ffill_close:
            new_close = new_close.vbt.ffill()
        new_close = new_close.values
        if self._open is not None:
            new_open = self.open.vbt.resample_apply(resampler, "first").values
        else:
            new_open = self._open
        if self._high is not None:
            new_high = self.high.vbt.resample_apply(resampler, "max").values
        else:
            new_high = self._high
        if self._low is not None:
            new_low = self.low.vbt.resample_apply(resampler, "min").values
        else:
            new_low = self._low
        new_order_records = self.orders.resample_records_arr(resampler)
        new_log_records = self.logs.resample_records_arr(resampler)
        if self._cash_deposits.size > 1 or self._cash_deposits.item() != 0:
            new_cash_deposits = self.get_cash_deposits(group_by=None if self.cash_sharing else False)
            new_cash_deposits = new_cash_deposits.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_deposits = new_cash_deposits.fillna(0.0)
            new_cash_deposits = new_cash_deposits.values
        else:
            new_cash_deposits = self._cash_deposits
        if self._cash_earnings.size > 1 or self._cash_earnings.item() != 0:
            new_cash_earnings = self.get_cash_earnings(group_by=False)
            new_cash_earnings = new_cash_earnings.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_earnings = new_cash_earnings.fillna(0.0)
            new_cash_earnings = new_cash_earnings.values
        else:
            new_cash_earnings = self._cash_earnings
        if self._bm_close is not None and not isinstance(self._bm_close, bool):
            new_bm_close = self.bm_close.vbt.resample_apply(resampler, "last")
            if fbfill_close:
                new_bm_close = new_bm_close.vbt.fbfill()
            elif ffill_close:
                new_bm_close = new_bm_close.vbt.ffill()
            new_bm_close = new_bm_close.values
        else:
            new_bm_close = self._bm_close
        if self._in_outputs is not None:
            new_in_outputs = self.resample_in_outputs(resampler, **resolve_dict(in_output_kwargs))
        else:
            new_in_outputs = None

        return self.replace(
            wrapper=new_wrapper,
            close=new_close,
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            log_records=new_log_records,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
        )

    # ############# Class methods ############# #

    @classmethod
    def from_orders(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        lock_cash: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_returns: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        skipna: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> PortfolioT:
        """Simulate portfolio from orders - size, price, fees, and other information.

        See `vectorbtpro.portfolio.nb.from_orders.simulate_from_orders_nb`.

        Args:
            close (array_like or Data): Latest asset price at each time step.
                Will broadcast.

                If an instance of `vectorbtpro.data.base.Data`, will extract the open, high,
                low, and close price.

                Used for calculating unrealized PnL and portfolio value.
            size (float or array_like): Size to order.
                See `vectorbtpro.portfolio.enums.Order.size`. Will broadcast.
            size_type (SizeType or array_like): See `vectorbtpro.portfolio.enums.SizeType` and
                `vectorbtpro.portfolio.enums.Order.size_type`. Will broadcast.
            direction (Direction or array_like): See `vectorbtpro.portfolio.enums.Direction` and
                `vectorbtpro.portfolio.enums.Order.direction`. Will broadcast.
            price (array_like of float): Order price.
                Will broadcast.

                See `vectorbtpro.portfolio.enums.Order.price`. Can be also provided as
                `vectorbtpro.portfolio.enums.PriceType`. Options `PriceType.NextOpen` and `PriceType.NextClose`
                are only applicable as single values, that is, they cannot be used inside arrays.
                In addition, they require the argument `from_ago` to be None.
            fees (float or array_like): Fees in percentage of the order value.
                See `vectorbtpro.portfolio.enums.Order.fees`. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
                See `vectorbtpro.portfolio.enums.Order.fixed_fees`. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price.
                See `vectorbtpro.portfolio.enums.Order.slippage`. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.
                See `vectorbtpro.portfolio.enums.Order.min_size`. Will broadcast.
            max_size (float or array_like): Maximum size for an order.
                See `vectorbtpro.portfolio.enums.Order.max_size`. Will broadcast.

                Will be partially filled if exceeded.
            size_granularity (float or array_like): Granularity of the size.
                See `vectorbtpro.portfolio.enums.Order.size_granularity`. Will broadcast.
            reject_prob (float or array_like): Order rejection probability.
                See `vectorbtpro.portfolio.enums.Order.reject_prob`. Will broadcast.
            price_area_vio_mode (PriceAreaVioMode or array_like): See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
                Will broadcast.
            lock_cash (bool or array_like): Whether to lock cash when shorting.
                See `vectorbtpro.portfolio.enums.Order.lock_cash`. Will broadcast.
            allow_partial (bool or array_like): Whether to allow partial fills.
                See `vectorbtpro.portfolio.enums.Order.allow_partial`. Will broadcast.

                Does not apply when size is `np.inf`.
            raise_reject (bool or array_like): Whether to raise an exception if order gets rejected.
                See `vectorbtpro.portfolio.enums.Order.raise_reject`. Will broadcast.
            log (bool or array_like): Whether to log orders.
                See `vectorbtpro.portfolio.enums.Order.log`. Will broadcast.
            val_price (array_like of float): Asset valuation price.
                Will broadcast.

                Can be also provided as `vectorbtpro.portfolio.enums.ValPriceType`.

                * Any `-np.inf` element is replaced by the latest valuation price
                    (`open` or the latest known valuation price if `ffill_val_price`).
                * Any `np.inf` element is replaced by the current order price.

                Used at the time of decision making to calculate value of each asset in the group,
                for example, to convert target value into target amount.

                !!! note
                    In contrast to `Portfolio.from_order_func`, order price is known beforehand (kind of),
                    thus `val_price` is set to the current order price (using `np.inf`) by default.
                    To valuate using previous close, set it in the settings to `-np.inf`.

                !!! note
                    Make sure to use timestamp for `val_price` that comes before timestamps of
                    all orders in the group with cash sharing (previous `close` for example),
                    otherwise you're cheating yourself.
            open (array_like of float): First asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            high (array_like of float): Highest asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            low (array_like of float): Lowest asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            init_cash (InitCashMode, float or array_like): Initial capital.

                By default, will broadcast to the final number of columns.
                But if cash sharing is enabled, will broadcast to the number of groups.
                See `vectorbtpro.portfolio.enums.InitCashMode` to find optimal initial cash.

                !!! note
                    Mode `InitCashMode.AutoAlign` is applied after the portfolio is initialized
                    to set the same initial cash for all columns/groups. Changing grouping
                    will change the initial cash, so be aware when indexing.
            init_position (float or array_like): Initial position.

                By default, will broadcast to the final number of columns.
            init_price (float or array_like): Initial position price.

                By default, will broadcast to the final number of columns.
            cash_deposits (float or array_like): Cash to be deposited/withdrawn at each timestamp.
                Will broadcast to the final shape. Must have the same number of columns as `init_cash`.

                Applied at the beginning of each timestamp.
            cash_earnings (float or array_like): Earnings in cash to be added at each timestamp.
                Will broadcast to the final shape.

                Applied at the end of each timestamp.
            cash_dividends (float or array_like): Dividends in cash to be added at each timestamp.
                Will broadcast to the final shape.

                Gets multiplied by the position and saved into `cash_earnings`.

                Applied at the end of each timestamp.
            cash_sharing (bool): Whether to share cash within the same group.

                If `group_by` is None and `cash_sharing` is True, `group_by` becomes True to form a single
                group with cash sharing.

                !!! warning
                    Introduces cross-asset dependencies.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel.
            from_ago (int or array_like): Take order information from a number of bars ago.
                Will broadcast.

                Negative numbers will be cast to positive to avoid the look-ahead bias. Defaults to 0.
                Remember to account of it if you're using a custom signal function!
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                Each value in this sequence must indicate the position of column in the group to
                call next. Processing of `call_seq` goes always from left to right.
                For example, `[2, 0, 1]` would first call column 'c', then 'a', and finally 'b'.

                Supported are multiple options:

                * Set to None to generate the default call sequence on the fly. Will create a
                    full array only if `attach_call_seq` is True.
                * Use `vectorbtpro.portfolio.enums.CallSeqType` to create a full array of a specific type.
                * Set to array to specify a custom call sequence.

                If `CallSeqType.Auto` selected, rearranges calls dynamically based on order value.
                Calculates value of all orders per row and group, and sorts them by this value.
                Sell orders will be executed first to release funds for buy orders.

                !!! warning
                    `CallSeqType.Auto` should be used with caution:

                    * It not only presumes that order prices are known beforehand, but also that
                        orders can be executed in arbitrary order and still retain their price.
                        In reality, this is hardly the case: after processing one asset, some time
                        has passed and the price for other assets might have already changed.
                    * Even if you're able to specify a slippage large enough to compensate for
                        this behavior, slippage itself should depend upon execution order.
                        This method doesn't let you do that.
                    * Orders in the same queue are executed regardless of whether previous orders
                        have been filled, which can leave them without required funds.

                    For more control, use `Portfolio.from_order_func`.
            attach_call_seq (bool): Whether to attach `call_seq` to the instance.

                Makes sense if you want to analyze the simulation order. Otherwise, just takes memory.
            ffill_val_price (bool): Whether to track valuation price only if it's known.

                Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp.
            update_value (bool): Whether to update group value after each filled order.
            fill_returns (bool): Whether to fill returns.

                The array will be avaiable as `returns` in in-outputs.
            max_orders (int): The max number of order records expected to be filled at each column.
                Defaults to the maximum number of non-NaN values across all columns of the size array.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            max_logs (int): The max number of log records expected to be filled at each column.
                Defaults to the maximum number of True values across all columns of the log array.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            skipna (bool): Whether to skip the rows where the size of all columns is NaN.

                Cannot work together with cash deposits, cash earnings, filling returns, and
                forward-filling the valuation price.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            group_by (any): Group columns. See `vectorbtpro.base.grouping.base.Grouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (any): See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            freq (any): Index frequency in case it cannot be parsed from `close`.
            bm_close (array_like): Latest benchmark price at each time step.
                Will broadcast.

                If not provided, will use `close`. If False, will not use any benchmark.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        !!! note
            When `call_seq` is not `CallSeqType.Auto`, at each timestamp, processing of the assets in
            a group goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why rebalancing
            cannot work properly in this setting: one has to specify percentages for all assets beforehand
            and then tweak the processing order to sell to-be-sold assets first in order to release funds
            for to-be-bought assets. This can be automatically done by using `CallSeqType.Auto`.

        !!! hint
            All broadcastable arguments can be set per frame, series, row, column, or element.

        Usage:
            * Buy 10 units each tick:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_orders(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Reverse each position by first closing it:

            ```pycon
            >>> size = [1, 0, -1, 0, 1]
            >>> pf = vbt.Portfolio.from_orders(close, size, size_type='targetpercent')

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * Regularly deposit cash at open and invest it within the same bar at close:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> cash_deposits = pd.Series([10., 0., 10., 0., 10.])
            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size=cash_deposits,  # invest the amount deposited
            ...     size_type='value',
            ...     cash_deposits=cash_deposits
            ... )

            >>> pf.cash
            0    100.0
            1    100.0
            2    100.0
            3    100.0
            4    100.0
            dtype: float64

            >>> pf.asset_flow
            0    10.000000
            1     0.000000
            2     3.333333
            3     0.000000
            4     2.000000
            dtype: float64
            ```

            * Equal-weighted portfolio as in `vectorbtpro.portfolio.nb.from_order_func.simulate_nb` example
            (it's more compact but has less control over execution):

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))
            >>> size = pd.Series(np.full(5, 1/3))  # each column 33.3%
            >>> size[1::2] = np.nan  # skip every second tick

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,  # acts both as reference and order price here
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',  # first sell then buy
            ...     group_by=True,  # one group
            ...     cash_sharing=True,  # assets share the same cash
            ...     fees=0.001, fixed_fees=1., slippage=0.001  # costs
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/simulate_nb_example.svg)

            * Test 10 random weight combinations:

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(
            ...     np.random.uniform(1, 10, size=(5, 3)),
            ...     columns=pd.Index(['a', 'b', 'c'], name='asset'))

            >>> # Generate random weight combinations
            >>> rand_weights = []
            >>> for i in range(10):
            ...     rand_weights.append(np.random.dirichlet(np.ones(close.shape[1]), size=1)[0])
            >>> rand_weights
            [array([0.15474873, 0.27706078, 0.5681905 ]),
             array([0.30468598, 0.18545189, 0.50986213]),
             array([0.15780486, 0.36292607, 0.47926907]),
             array([0.25697713, 0.64902589, 0.09399698]),
             array([0.43310548, 0.53836359, 0.02853093]),
             array([0.78628605, 0.15716865, 0.0565453 ]),
             array([0.37186671, 0.42150531, 0.20662798]),
             array([0.22441579, 0.06348919, 0.71209502]),
             array([0.41619664, 0.09338007, 0.49042329]),
             array([0.01279537, 0.87770864, 0.10949599])]

            >>> # Bring close and rand_weights to the same shape
            >>> rand_weights = np.concatenate(rand_weights)
            >>> close = close.vbt.tile(10, keys=pd.Index(np.arange(10), name='weights_vector'))
            >>> size = vbt.broadcast_to(weights, close).copy()
            >>> size[1::2] = np.nan
            >>> size
            weights_vector                            0  ...                               9
            asset                  a         b        c  ...           a         b         c
            0               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            1                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            2               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            3                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            4               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496

            [5 rows x 30 columns]

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',
            ...     group_by='weights_vector',  # group by column level
            ...     cash_sharing=True,
            ...     fees=0.001, fixed_fees=1., slippage=0.001
            ... )

            >>> pf.total_return
            weights_vector
            0   -0.294372
            1    0.139207
            2   -0.281739
            3    0.041242
            4    0.467566
            5    0.829925
            6    0.320672
            7   -0.087452
            8    0.376681
            9   -0.702773
            Name: total_return, dtype: float64
            ```
        """
        # Get defaults
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if isinstance(close, Data):
            data = close
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            if open is None:
                open = data.open
            if high is None:
                high = data.high
            if low is None:
                low = data.low
        if open is None:
            open_none = True
            open = np.nan
        else:
            open_none = False
        if high is None:
            high_none = True
            high = np.nan
        else:
            high_none = False
        if low is None:
            low_none = True
            low = np.nan
        else:
            low_none = False

        if size is None:
            size = portfolio_cfg["size"]
        if size_type is None:
            size_type = portfolio_cfg["size_type"]
        if direction is None:
            direction = portfolio_cfg["direction"]
        if price is None:
            price = portfolio_cfg["price"]
        if isinstance(price, str):
            price = map_enum_fields(price, PriceType)
        if isinstance(price, (int, float)):
            if price in (-1, -2):
                if from_ago is not None:
                    raise ValueError("Price of next open/close and from_ago cannot be used simultaneously")
                if price == -1:
                    price = -np.inf
                if price == -2:
                    price = np.inf
                from_ago = 1
        if size is None:
            size = portfolio_cfg["size"]
        if fees is None:
            fees = portfolio_cfg["fees"]
        if fixed_fees is None:
            fixed_fees = portfolio_cfg["fixed_fees"]
        if slippage is None:
            slippage = portfolio_cfg["slippage"]
        if min_size is None:
            min_size = portfolio_cfg["min_size"]
        if max_size is None:
            max_size = portfolio_cfg["max_size"]
        if size_granularity is None:
            size_granularity = portfolio_cfg["size_granularity"]
        if reject_prob is None:
            reject_prob = portfolio_cfg["reject_prob"]
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg["price_area_vio_mode"]
        if lock_cash is None:
            lock_cash = portfolio_cfg["lock_cash"]
        if allow_partial is None:
            allow_partial = portfolio_cfg["allow_partial"]
        if raise_reject is None:
            raise_reject = portfolio_cfg["raise_reject"]
        if log is None:
            log = portfolio_cfg["log"]
        if val_price is None:
            val_price = portfolio_cfg["val_price"]
        if init_cash is None:
            init_cash = portfolio_cfg["init_cash"]
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if checks.is_int(init_cash) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg["init_position"]
        if init_price is None:
            init_price = portfolio_cfg["init_price"]
        if cash_deposits is None:
            cash_deposits = portfolio_cfg["cash_deposits"]
        if cash_earnings is None:
            cash_earnings = portfolio_cfg["cash_earnings"]
        if cash_dividends is None:
            cash_dividends = portfolio_cfg["cash_dividends"]
        if cash_sharing is None:
            cash_sharing = portfolio_cfg["cash_sharing"]
        if cash_sharing and group_by is None:
            group_by = True
        if from_ago is None:
            from_ago = portfolio_cfg["from_ago"]
        if call_seq is None:
            call_seq = portfolio_cfg["call_seq"]
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if checks.is_int(call_seq):
            if call_seq == CallSeqType.Auto:
                auto_call_seq = True
                call_seq = None
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg["attach_call_seq"]
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg["ffill_val_price"]
        if update_value is None:
            update_value = portfolio_cfg["update_value"]
        if fill_returns is None:
            fill_returns = portfolio_cfg["fill_returns"]
        if skipna is None:
            skipna = portfolio_cfg["skipna"]
        if seed is None:
            seed = portfolio_cfg["seed"]
        if seed is not None:
            set_seed(seed)
        if group_by is None:
            group_by = portfolio_cfg["group_by"]
        if freq is None:
            freq = portfolio_cfg["freq"]
        broadcast_kwargs = merge_dicts(portfolio_cfg["broadcast_kwargs"], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get("require_kwargs", {})
        if bm_close is None:
            bm_close = portfolio_cfg["bm_close"]

        # Prepare the simulation
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        broadcastable_args = dict(
            cash_earnings=cash_earnings,
            cash_dividends=cash_dividends,
            size=size,
            price=price,
            size_type=size_type,
            direction=direction,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            reject_prob=reject_prob,
            price_area_vio_mode=price_area_vio_mode,
            lock_cash=lock_cash,
            allow_partial=allow_partial,
            raise_reject=raise_reject,
            log=log,
            val_price=val_price,
            open=open,
            high=high,
            low=low,
            close=close,
            from_ago=from_ago,
        )
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args["bm_close"] = bm_close
        else:
            broadcastable_args["bm_close"] = None

        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=True,
                reindex_kwargs=dict(
                    cash_earnings=dict(fill_value=0.0),
                    cash_dividends=dict(fill_value=0.0),
                    size=dict(fill_value=np.nan),
                    price=dict(fill_value=np.nan),
                    size_type=dict(fill_value=SizeType.Amount),
                    direction=dict(fill_value=Direction.Both),
                    fees=dict(fill_value=0.0),
                    fixed_fees=dict(fill_value=0.0),
                    slippage=dict(fill_value=0.0),
                    min_size=dict(fill_value=np.nan),
                    max_size=dict(fill_value=np.nan),
                    size_granularity=dict(fill_value=np.nan),
                    reject_prob=dict(fill_value=0.0),
                    price_area_vio_mode=dict(fill_value=PriceAreaVioMode.Ignore),
                    lock_cash=dict(fill_value=False),
                    allow_partial=dict(fill_value=True),
                    raise_reject=dict(fill_value=False),
                    log=dict(fill_value=False),
                    val_price=dict(fill_value=np.nan),
                    open=dict(fill_value=np.nan),
                    high=dict(fill_value=np.nan),
                    low=dict(fill_value=np.nan),
                    close=dict(fill_value=np.nan),
                    bm_close=dict(fill_value=np.nan),
                    from_ago=dict(fill_value=0),
                ),
                wrapper_kwargs=dict(
                    freq=freq,
                    group_by=group_by,
                ),
            ),
            broadcast_kwargs,
        )
        broadcasted_args, wrapper = broadcast(broadcastable_args, return_wrapper=True, **broadcast_kwargs)
        if not wrapper.group_select and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        cash_earnings = broadcasted_args.pop("cash_earnings")
        cash_dividends = broadcasted_args.pop("cash_dividends")
        target_shape_2d = wrapper.shape_2d

        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        init_price = np.require(np.broadcast_to(init_price, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            cash_deposits,
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            keep_flex=True,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=require_kwargs,
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if call_seq is None and attach_call_seq:
            call_seq = CallSeqType.Default
        if call_seq is not None:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
            else:
                call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if max_orders is None:
            _size = broadcasted_args["size"]
            if _size.size == 1:
                max_orders = target_shape_2d[0] * int(not np.isnan(_size.item(0)))
            else:
                if _size.shape[0] == 1 and target_shape_2d[0] > 1:
                    max_orders = target_shape_2d[0] * int(np.any(~np.isnan(_size)))
                else:
                    max_orders = int(np.max(np.sum(~np.isnan(_size), axis=0)))
        if max_logs is None:
            _log = broadcasted_args["log"]
            if _log.size == 1:
                max_logs = target_shape_2d[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and target_shape_2d[0] > 1:
                    max_logs = target_shape_2d[0] * int(np.any(_log))
                else:
                    max_logs = int(np.max(np.sum(_log, axis=0)))

        # Convert strings to numbers
        broadcasted_args["price"] = map_enum_fields(broadcasted_args["price"], PriceType, ignore_type=(int, float))
        broadcasted_args["size_type"] = map_enum_fields(broadcasted_args["size_type"], SizeType)
        broadcasted_args["direction"] = map_enum_fields(broadcasted_args["direction"], Direction)
        broadcasted_args["price_area_vio_mode"] = map_enum_fields(
            broadcasted_args["price_area_vio_mode"],
            PriceAreaVioMode,
        )
        broadcasted_args["val_price"] = map_enum_fields(
            broadcasted_args["val_price"],
            ValPriceType,
            ignore_type=(int, float),
        )

        # Check data types
        checks.assert_subdtype(cs_group_lens, np.integer)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(init_price, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(cash_dividends, np.number)
        checks.assert_subdtype(broadcasted_args["size"], np.number)
        checks.assert_subdtype(broadcasted_args["price"], np.number)
        checks.assert_subdtype(broadcasted_args["size_type"], np.integer)
        checks.assert_subdtype(broadcasted_args["direction"], np.integer)
        checks.assert_subdtype(broadcasted_args["fees"], np.number)
        checks.assert_subdtype(broadcasted_args["fixed_fees"], np.number)
        checks.assert_subdtype(broadcasted_args["slippage"], np.number)
        checks.assert_subdtype(broadcasted_args["min_size"], np.number)
        checks.assert_subdtype(broadcasted_args["max_size"], np.number)
        checks.assert_subdtype(broadcasted_args["size_granularity"], np.number)
        checks.assert_subdtype(broadcasted_args["reject_prob"], np.number)
        checks.assert_subdtype(broadcasted_args["price_area_vio_mode"], np.integer)
        checks.assert_subdtype(broadcasted_args["lock_cash"], np.bool_)
        checks.assert_subdtype(broadcasted_args["allow_partial"], np.bool_)
        checks.assert_subdtype(broadcasted_args["raise_reject"], np.bool_)
        checks.assert_subdtype(broadcasted_args["log"], np.bool_)
        checks.assert_subdtype(broadcasted_args["val_price"], np.number)
        checks.assert_subdtype(broadcasted_args["open"], np.number)
        checks.assert_subdtype(broadcasted_args["high"], np.number)
        checks.assert_subdtype(broadcasted_args["low"], np.number)
        checks.assert_subdtype(broadcasted_args["close"], np.number)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args["bm_close"], np.number)
        checks.assert_subdtype(broadcasted_args["from_ago"], np.integer)

        # Remove arguments
        bm_close = broadcasted_args.pop("bm_close", None)

        # Perform the simulation
        func = jit_reg.resolve_option(nb.simulate_from_orders_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        sim_out = func(
            target_shape=target_shape_2d,
            group_lens=cs_group_lens,  # group only if cash sharing is enabled to speed up
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            cash_dividends=cash_dividends,
            **broadcasted_args,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_returns=fill_returns,
            max_orders=max_orders,
            max_logs=max_logs,
            skipna=skipna,
        )

        # Create an instance
        return cls(
            wrapper,
            broadcasted_args["close"],
            sim_out.order_records,
            open=broadcasted_args["open"] if not open_none else None,
            high=broadcasted_args["high"] if not high_none else None,
            low=broadcasted_args["low"] if not low_none else None,
            log_records=sim_out.log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=sim_out.cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs,
        )

    @classmethod
    def from_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        entries: tp.Optional[tp.ArrayLike] = None,
        exits: tp.Optional[tp.ArrayLike] = None,
        short_entries: tp.Optional[tp.ArrayLike] = None,
        short_exits: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        adjust_func_nb: nb.AdjustFuncT = nb.no_adjust_func_nb,
        adjust_args: tp.Args = (),
        signal_func_nb: nb.SignalFuncT = nb.no_signal_func_nb,
        signal_args: tp.ArgsLike = (),
        post_segment_func_nb: nb.SignalFuncT = nb.no_post_func_nb,
        post_segment_args: tp.ArgsLike = (),
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        lock_cash: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        accumulate: tp.Optional[tp.ArrayLike] = None,
        upon_long_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_short_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_dir_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opposite_entry: tp.Optional[tp.ArrayLike] = None,
        order_type: tp.Optional[tp.ArrayLike] = None,
        limit_delta: tp.Optional[tp.ArrayLike] = None,
        limit_tif: tp.Optional[tp.ArrayLike] = None,
        limit_expiry: tp.Optional[tp.ArrayLike] = None,
        limit_reverse: tp.Optional[tp.ArrayLike] = None,
        upon_adj_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        use_stops: tp.Optional[bool] = None,
        sl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_th: tp.Optional[tp.ArrayLike] = None,
        tp_stop: tp.Optional[tp.ArrayLike] = None,
        stop_entry_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_type: tp.Optional[tp.ArrayLike] = None,
        stop_order_type: tp.Optional[tp.ArrayLike] = None,
        stop_limit_delta: tp.Optional[tp.ArrayLike] = None,
        upon_stop_update: tp.Optional[tp.ArrayLike] = None,
        upon_adj_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        delta_format: tp.Optional[tp.ArrayLike] = None,
        time_delta_format: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_returns: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> PortfolioT:
        """Simulate portfolio from entry and exit signals.

        See `vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb`.

        You have three options to provide signals:

        1. `entries` and `exits`:
            Uses `vectorbtpro.portfolio.nb.from_signals.dir_enex_signal_func_nb` as `signal_func_nb`.
        2. `entries` (acting as long), `exits` (acting as long), `short_entries`, and `short_exits`:
            Uses `vectorbtpro.portfolio.nb.from_signals.ls_enex_signal_func_nb` as `signal_func_nb`.
        3. `signal_func_nb` and `signal_args`: Custom signal function that returns direction-aware signals.

        Args:
            close (array_like or Data): See `Portfolio.from_orders`.
            entries (array_like of bool): Boolean array of entry signals.
                Defaults to True if all other signal arrays are not set, otherwise False. Will broadcast.

                * If `short_entries` and `short_exits` are not set: Acts as a long signal if `direction`
                    is 'all' or 'longonly', otherwise short.
                * If `short_entries` or `short_exits` are set: Acts as `long_entries`.
            exits (array_like of bool): Boolean array of exit signals.
                Defaults to False. Will broadcast.

                * If `short_entries` and `short_exits` are not set: Acts as a short signal if `direction`
                    is 'all' or 'longonly', otherwise long.
                * If `short_entries` or `short_exits` are set: Acts as `long_exits`.
            short_entries (array_like of bool): Boolean array of short entry signals.
                Defaults to False. Will broadcast.
            short_exits (array_like of bool): Boolean array of short exit signals.
                Defaults to False. Will broadcast.
            direction (Direction or array_like): See `Portfolio.from_orders`.

                Takes only effect if `short_entries` and `short_exits` are not set.
            adjust_func_nb (callable): User-defined function to adjust the current simulation state.
                Defaults to `vectorbtpro.portfolio.nb.from_signals.no_adjust_func_nb`.

                Passed as argument to `vectorbtpro.portfolio.nb.from_signals.dir_enex_signal_func_nb`
                and `vectorbtpro.portfolio.nb.from_signals.ls_enex_signal_func_nb`. Has no effect
                when using other signal functions.
            adjust_args (tuple): Packed arguments passed to `adjust_func_nb`.
                Defaults to `()`.
            signal_func_nb (callable): Function called to generate signals.

                See `vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb`.
            signal_args (tuple): Packed arguments passed to `signal_func_nb`.
                Defaults to `()`.
            post_segment_func_nb (callable): Post-segment function.

                See `vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb`.
            post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
                Defaults to `()`.
            size (float or array_like): See `Portfolio.from_orders`.

                !!! note
                    Negative size is not allowed. You must express direction using signals.
            size_type (SizeType or array_like): See `Portfolio.from_orders`.

                Only `SizeType.Amount`, `SizeType.Value`, `SizeType.Percent(100)`, and
                `SizeType.ValuePercent(100)` are supported. Other modes such as target percentage
                are not compatible with signals since their logic may contradict the direction of the signal.

                !!! note
                    `SizeType.Percent(100)` does not support position reversal. Switch to a single
                    direction or use `OppositeEntryMode.Close` to close the position first.

                See warning in `Portfolio.from_orders`.
            price (array_like of float): See `Portfolio.from_orders`.
            fees (float or array_like): See `Portfolio.from_orders`.
            fixed_fees (float or array_like): See `Portfolio.from_orders`.
            slippage (float or array_like): See `Portfolio.from_orders`.
            min_size (float or array_like): See `Portfolio.from_orders`.
            max_size (float or array_like): See `Portfolio.from_orders`.

                Will be partially filled if exceeded. You might not be able to properly close
                the position if accumulation is enabled and `max_size` is too low.
            size_granularity (float or array_like): See `Portfolio.from_orders`.
            reject_prob (float or array_like): See `Portfolio.from_orders`.
            price_area_vio_mode (PriceAreaVioMode or array_like): See `Portfolio.from_orders`.
            lock_cash (bool or array_like): See `Portfolio.from_orders`.
            allow_partial (bool or array_like): See `Portfolio.from_orders`.
            raise_reject (bool or array_like): See `Portfolio.from_orders`.
            log (bool or array_like): See `Portfolio.from_orders`.
            val_price (array_like of float): See `Portfolio.from_orders`.
            accumulate (bool, AccumulationMode or array_like): See `vectorbtpro.portfolio.enums.AccumulationMode`.
                If True, becomes 'both'. If False, becomes 'disabled'. Will broadcast.

                When enabled, `Portfolio.from_signals` behaves similarly to `Portfolio.from_orders`.
            upon_long_conflict (ConflictMode or array_like): Conflict mode for long signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_short_conflict (ConflictMode or array_like): Conflict mode for short signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_dir_conflict (DirectionConflictMode or array_like): See `vectorbtpro.portfolio.enums.DirectionConflictMode`.
                Will broadcast.
            upon_opposite_entry (OppositeEntryMode or array_like): See `vectorbtpro.portfolio.enums.OppositeEntryMode`.
                Will broadcast.
            order_type (OrderType or array_like): See `vectorbtpro.portfolio.enums.OrderType`.

                Only one active limit order is allowed at a time.
            limit_delta (float or array_like): Delta from `price` to build the limit price.
                Will broadcast.

                If NaN, `price` becomes the limit price. Otherwise, applied on top of `price` depending
                on the current direction: if the direction-aware size is positive (= buying), a positive delta
                will decrease the limit price; if the direction-aware size is negative (= selling), a positive delta
                will increase the limit price. Delta can be negative.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            limit_tif (frequency_like or array_like): Time in force for limit signals.
                Will broadcast.

                Any frequency-like object is converted using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
                Any array must either contain timedeltas or integers, and will be cast into integer format
                after broadcasting. If the object provided is of data type `object`, will be converted
                to timedelta automatically.

                Measured in the distance after the open time of the signal bar. If the expiration time happens
                in the middle of the current bar, we pessimistically assume that the order has been expired.
                The check is performed at the beginning of the bar, and the first check is performed at the
                next bar after the signal. For example, if the format is `TimeDeltaFormat.Rows`, 0 or 1 means
                the order must execute at the same bar or not at all; 2 means the order must execute at the
                same or next bar or not at all.

                Set an element to `-1` to disable. Use `time_delta_format` to specify the format.
            limit_expiry (frequency_like, datetime_like, or array_like): Expiration time.
                Will broadcast.

                Any frequency-like object is used to build a period index, such that each timestamp in the original
                index is pointing to the timestamp where the period ends. For example, providing "d" will
                make any limit order expire on the next day. Any array must either contain timestamps or integers
                (not timedeltas!), and will be cast into integer format after broadcasting. If the object
                provided is of data type `object`, will be converted to datetime and its timezone will
                be removed automatically (as done on the index).

                Behaves in a similar way as `limit_tif`.

                Set an element to `-1` or `pd.Timestamp.max` to disable. Use `time_delta_format` to specify the format.
            limit_reverse (bool or array_like): Whether to reverse the price hit detection.
                Will broadcast.

                If True, a buy/sell limit price will be checked against high/low (not low/high).
                Also, the limit delta will be applied above/below (not below/above) the initial price.
            upon_adj_limit_conflict (PendingConflictMode or array_like): Conflict mode for limit and user-defined
                signals of adjacent sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            upon_opp_limit_conflict (PendingConflictMode or array_like): Conflict mode for limit and user-defined
                signals of opposite sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            use_stops (bool): Whether to use stops.
                Defaults to None, which becomes True if any of the stops are not NaN or
                the adjustment function is not the default one.

                Disable this to make simulation a bit faster for simple use cases.
            sl_stop (array_like of float): Stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tsl_stop (array_like of float): Trailing stop loss for the trailing stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tsl_th (array_like of float): Take profit threshold for the trailing stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tp_stop (array_like of float): Take profit.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            stop_entry_price (StopEntryPrice or array_like): See `vectorbtpro.portfolio.enums.StopEntryPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry. If a positive value is provided,
                used directly as a price, otherwise used as an enumerated value.
            stop_exit_price (StopExitPrice or array_like): See `vectorbtpro.portfolio.enums.StopExitPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry. If a positive value is provided,
                used directly as a price, otherwise used as an enumerated value.
            stop_exit_type (StopExitType or array_like): See `vectorbtpro.portfolio.enums.StopExitType`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry.
            stop_order_type (OrderType or array_like): Similar to `order_type` but for stop orders.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry.
            stop_limit_delta (float or array_like): Similar to `limit_delta` but for stop orders.
                Will broadcast.
            upon_stop_update (StopUpdateMode or array_like): See `vectorbtpro.portfolio.enums.StopUpdateMode`.
                Will broadcast.

                Only has effect if accumulation is enabled.

                If provided on per-element basis, gets applied upon repeated entry.
            upon_adj_stop_conflict (PendingConflictMode or array_like): Conflict mode for stop and user-defined
                signals of adjacent sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            upon_opp_stop_conflict (PendingConflictMode or array_like): Conflict mode for stop and user-defined
                signals of opposite sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            delta_format (DeltaFormat or array_like): See `vectorbtpro.portfolio.enums.DeltaFormat`.
                Will broadcast.
            time_delta_format (TimeDeltaFormat or array_like): See `vectorbtpro.portfolio.enums.TimeDeltaFormat`.
                Will broadcast.
            open (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` gets replaced by `close`.
            high (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the maximum out of `open` and `close`.
            low (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the minimum out of `open` and `close`.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            init_price (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_dividends (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): See `Portfolio.from_orders`.
            from_ago (int or array_like): See `Portfolio.from_orders`.

                Take effect only for user-defined signals, not for stop signals.
            call_seq (CallSeqType or array_like): See `Portfolio.from_orders`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            ffill_val_price (bool): See `Portfolio.from_orders`.
            update_value (bool): See `Portfolio.from_orders`.
            fill_returns (bool): See `Portfolio.from_orders`.
            max_orders (int): See `Portfolio.from_orders`.
            max_logs (int): See `Portfolio.from_orders`.
            in_outputs (mapping_like): Mapping with in-output objects. Only for flexible mode.

                Will be available via `Portfolio.in_outputs` as a named tuple.

                To substitute `Portfolio` attributes, provide already broadcasted and grouped objects,
                for example, by using `broadcast_named_args` and templates. Also see
                `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.

                When chunking, make sure to provide the chunk taking specification and the merging function.
                See `vectorbtpro.portfolio.chunking.merge_sim_outs`.

                !!! note
                    When using Numba below 0.54, `in_outputs` cannot be a mapping, but must be a named tuple
                    defined globally so Numba can introspect its attributes for pickling.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): Dictionary with named arguments to broadcast.

                You can then pass argument names wrapped with `vectorbtpro.utils.template.Rep`
                and this method will substitute them by their corresponding broadcasted objects.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_context (mapping): Mapping to replace templates in arguments.
            jitted (any): See `Portfolio.from_orders`.
            chunked (any): See `Portfolio.from_orders`.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        Also see notes and hints for `Portfolio.from_orders`.

        Usage:
            * By default, if all signal arrays are None, `entries` becomes True,
            which opens a position at the very first tick and does nothing else:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_signals(close, size=1)
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * Entry opens long, exit closes long:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='longonly'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=pd.Series([False, False, True, True, True]),  # long_exits
            ...     short_entries=False,
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64
            ```

            Notice how both `short_entries` and `short_exits` are provided as constants - as any other
            broadcastable argument, they are treated as arrays where each element is False.

            * Entry opens short, exit closes short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='shortonly'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=False,  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([True, True, True, False, False]),
            ...     short_exits=pd.Series([False, False, True, True, True]),
            ...     size=1
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64
            ```

            * Entry opens long and closes short, exit closes long and opens short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([False, False, True, True, True]),
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64
            ```

            * More complex signal combinations are best expressed using direction-aware arrays.
            For example, ignore opposite signals as long as the current position is open:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries      =pd.Series([True, False, False, False, False]),  # long_entries
            ...     exits        =pd.Series([False, False, True, False, False]),  # long_exits
            ...     short_entries=pd.Series([False, True, False, True, False]),
            ...     short_exits  =pd.Series([False, False, False, False, True]),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            * First opposite signal closes the position, second one opens a new position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both',
            ...     upon_opposite_entry='close'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * If both long entry and exit signals are True (a signal conflict), choose exit:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='longonly',
            ...     upon_long_conflict='exit')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * If both long entry and short entry signal are True (a direction conflict), choose short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     upon_dir_conflict='short')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -2.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            !!! note
                Remember that when direction is set to 'both', entries become `long_entries` and exits become
                `short_entries`, so this becomes a conflict of directions rather than signals.

            * If there are both signal and direction conflicts:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=True,  # long_entries
            ...     exits=True,  # long_exits
            ...     short_entries=True,
            ...     short_exits=True,
            ...     size=1,
            ...     upon_long_conflict='entry',
            ...     upon_short_conflict='entry',
            ...     upon_dir_conflict='short'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * Turn on accumulation of signals. Entry means long order, exit means short order
            (acts similar to `from_orders`):

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate=True)
            >>> pf.asset_flow
            0    1.0
            1    1.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * Allow increasing a position (of any direction), deny decreasing a position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate='addonly')
            >>> pf.asset_flow
            0    1.0  << open a long position
            1    1.0  << add to the position
            2    0.0
            3   -3.0  << close and open a short position
            4   -1.0  << add to the position
            dtype: float64
            ```

            * Test multiple parameters via regular broadcasting:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=[list(Direction)],
            ...     broadcast_kwargs=dict(columns_from=pd.Index(vbt.pf_enums.Direction._fields, name='direction')))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            * Test multiple parameters via `vectorbtpro.base.reshaping.BCO`:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=vbt.BCO(Direction, product=True))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            * Set risk/reward ratio by passing trailing stop loss and take profit thresholds:

            ```pycon
            >>> close = pd.Series([10, 11, 12, 11, 10, 9])
            >>> entries = pd.Series([True, False, False, False, False, False])
            >>> exits = pd.Series([False, False, False, False, False, True])
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.2)  # take profit hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.3)  # trailing stop loss hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4   -10.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=np.inf, tp_stop=np.inf)  # nothing hit, exit as usual
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4     0.0
            5   -10.0
            dtype: float64
            ```

            * Test different stop combinations:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=pd.Index([0.1, 0.2]),
            ...     tp_stop=pd.Index([0.2, 0.3])
            ... )
            >>> pf.asset_flow
            tsl_stop   0.1         0.2
            tp_stop    0.2   0.3   0.2   0.3
            0         10.0  10.0  10.0  10.0
            1          0.0   0.0   0.0   0.0
            2        -10.0   0.0 -10.0   0.0
            3          0.0   0.0   0.0   0.0
            4          0.0 -10.0   0.0   0.0
            5          0.0   0.0   0.0 -10.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            * We can implement our own stop loss or take profit, or adjust the existing one at each time step.
            Let's implement [stepped stop-loss](https://www.freqtrade.io/en/stable/strategy-advanced/#stepped-stoploss):

            ```pycon
            >>> @njit
            ... def adjust_func_nb(c):
            ...     val_price_now = c.last_val_price[c.col]
            ...     tsl_init_price = c.last_tsl_info["init_price"][c.col]
            ...     current_profit = (val_price_now - tsl_init_price) / tsl_init_price
            ...     if current_profit >= 0.40:
            ...         c.last_tsl_info["stop"][c.col] = 0.25
            ...     elif current_profit >= 0.25:
            ...         c.last_tsl_info["stop"][c.col] = 0.15
            ...     elif current_profit >= 0.20:
            ...         c.last_tsl_info["stop"][c.col] = 0.07

            >>> close = pd.Series([10, 11, 12, 11, 10])
            >>> pf = vbt.Portfolio.from_signals(close, adjust_func_nb=adjust_func_nb)
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0  # 7% from 12 hit
            4    11.16
            dtype: float64
            ```

            * Sometimes there is a need to provide or transform signals dynamically. For this, we can implement
            a custom signal function `signal_func_nb`. For example, let's implement a signal function that
            takes two numerical arrays - long and short one - and transforms them into 4 direction-aware boolean
            arrays that vectorbt understands:

            ```pycon
            >>> @njit
            ... def signal_func_nb(c, long_num_arr, short_num_arr):
            ...     long_num = vbt.pf_nb.select_nb(c, long_num_arr)
            ...     short_num = vbt.pf_nb.select_nb(c, short_num_arr)
            ...     is_long_entry = long_num > 0
            ...     is_long_exit = long_num < 0
            ...     is_short_entry = short_num > 0
            ...     is_short_exit = short_num < 0
            ...     return is_long_entry, is_long_exit, is_short_entry, is_short_exit

            >>> pf = vbt.Portfolio.from_signals(
            ...     pd.Series([1, 2, 3, 4, 5]),
            ...     signal_func_nb=signal_func_nb,
            ...     signal_args=(vbt.Rep('long_num_arr'), vbt.Rep('short_num_arr')),
            ...     broadcast_named_args=dict(
            ...         long_num_arr=pd.Series([1, 0, -1, 0, 0]),
            ...         short_num_arr=pd.Series([0, 1, 0, 1, -1])
            ...     ),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            Passing both arrays as `broadcast_named_args` broadcasts them internally as any other array,
            so we don't have to worry about their dimensions every time we change our data.
        """
        # Get defaults
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if isinstance(close, Data):
            data = close
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            if open is None:
                open = data.open
            if high is None:
                high = data.high
            if low is None:
                low = data.low
        if open is None:
            open_none = True
            open = np.nan
        else:
            open_none = False
        if high is None:
            high_none = True
            high = np.nan
        else:
            high_none = False
        if low is None:
            low_none = True
            low = np.nan
        else:
            low_none = False

        flexible_mode = (
            (adjust_func_nb is not nb.no_adjust_func_nb)
            or (signal_func_nb is not nb.no_signal_func_nb)
            or (post_segment_func_nb is not nb.no_post_func_nb)
        )
        ls_mode = short_entries is not None or short_exits is not None
        signal_func_mode = signal_func_nb is not nb.no_signal_func_nb
        if (entries is not None or exits is not None or ls_mode) and signal_func_mode:
            raise ValueError("Either any of the signal arrays or signal_func_nb must be provided, not both")
        if entries is None:
            entries = False
        if exits is None:
            exits = False
        if short_entries is None:
            short_entries = False
        if short_exits is None:
            short_exits = False
        if direction is not None and ls_mode:
            warnings.warn("direction has no effect if short_entries and short_exits are set", stacklevel=2)
        if direction is None:
            direction = portfolio_cfg["signal_direction"]
        if signal_func_nb is nb.no_signal_func_nb:
            if ls_mode:
                signal_func_nb = nb.ls_enex_signal_func_nb
            else:
                signal_func_nb = nb.dir_enex_signal_func_nb
        if size is None:
            size = portfolio_cfg["size"]
        if size_type is None:
            size_type = portfolio_cfg["size_type"]
        if price is None:
            price = portfolio_cfg["price"]
        if isinstance(price, str):
            price = map_enum_fields(price, PriceType)
        if isinstance(price, (int, float)):
            if price in (-1, -2):
                if from_ago is not None:
                    raise ValueError("Price of next open/close and from_ago cannot be used simultaneously")
                if price == -1:
                    price = -np.inf
                if price == -2:
                    price = np.inf
                from_ago = 1
        if fees is None:
            fees = portfolio_cfg["fees"]
        if fixed_fees is None:
            fixed_fees = portfolio_cfg["fixed_fees"]
        if slippage is None:
            slippage = portfolio_cfg["slippage"]
        if min_size is None:
            min_size = portfolio_cfg["min_size"]
        if max_size is None:
            max_size = portfolio_cfg["max_size"]
        if size_granularity is None:
            size_granularity = portfolio_cfg["size_granularity"]
        if reject_prob is None:
            reject_prob = portfolio_cfg["reject_prob"]
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg["price_area_vio_mode"]
        if lock_cash is None:
            lock_cash = portfolio_cfg["lock_cash"]
        if allow_partial is None:
            allow_partial = portfolio_cfg["allow_partial"]
        if raise_reject is None:
            raise_reject = portfolio_cfg["raise_reject"]
        if log is None:
            log = portfolio_cfg["log"]
        if val_price is None:
            val_price = portfolio_cfg["val_price"]
        if accumulate is None:
            accumulate = portfolio_cfg["accumulate"]
        if upon_long_conflict is None:
            upon_long_conflict = portfolio_cfg["upon_long_conflict"]
        if upon_short_conflict is None:
            upon_short_conflict = portfolio_cfg["upon_short_conflict"]
        if upon_dir_conflict is None:
            upon_dir_conflict = portfolio_cfg["upon_dir_conflict"]
        if upon_opposite_entry is None:
            upon_opposite_entry = portfolio_cfg["upon_opposite_entry"]
        if order_type is None:
            order_type = portfolio_cfg["order_type"]
        if limit_delta is None:
            limit_delta = portfolio_cfg["limit_delta"]
        if limit_tif is None:
            limit_tif = portfolio_cfg["limit_tif"]
        if isinstance(limit_tif, (str, timedelta, pd.DateOffset, pd.Timedelta)):
            limit_tif = freq_to_timedelta64(limit_tif)
        if limit_expiry is None:
            limit_expiry = portfolio_cfg["limit_expiry"]
        if isinstance(limit_expiry, (str, timedelta, pd.DateOffset, pd.Timedelta)):
            limit_expiry = RepEval(
                "wrapper.get_period_ns_index(limit_expiry)[:, None]",
                context=dict(limit_expiry=limit_expiry),
            )
        if limit_reverse is None:
            limit_reverse = portfolio_cfg["limit_reverse"]
        if upon_adj_limit_conflict is None:
            upon_adj_limit_conflict = portfolio_cfg["upon_adj_limit_conflict"]
        if upon_opp_limit_conflict is None:
            upon_opp_limit_conflict = portfolio_cfg["upon_opp_limit_conflict"]
        if sl_stop is None:
            sl_stop = portfolio_cfg["sl_stop"]
        if tsl_stop is None:
            tsl_stop = portfolio_cfg["tsl_stop"]
        if tsl_th is None:
            tsl_th = portfolio_cfg["tsl_th"]
        if tp_stop is None:
            tp_stop = portfolio_cfg["tp_stop"]
        if use_stops is None:
            use_stops = portfolio_cfg["use_stops"]
        if stop_entry_price is None:
            stop_entry_price = portfolio_cfg["stop_entry_price"]
        if stop_exit_price is None:
            stop_exit_price = portfolio_cfg["stop_exit_price"]
        if stop_exit_type is None:
            stop_exit_type = portfolio_cfg["stop_exit_type"]
        if stop_order_type is None:
            stop_order_type = portfolio_cfg["stop_order_type"]
        if stop_limit_delta is None:
            stop_limit_delta = portfolio_cfg["stop_limit_delta"]
        if upon_stop_update is None:
            upon_stop_update = portfolio_cfg["upon_stop_update"]
        if upon_adj_stop_conflict is None:
            upon_adj_stop_conflict = portfolio_cfg["upon_adj_stop_conflict"]
        if upon_opp_stop_conflict is None:
            upon_opp_stop_conflict = portfolio_cfg["upon_opp_stop_conflict"]
        if delta_format is None:
            delta_format = portfolio_cfg["delta_format"]
        if time_delta_format is None:
            time_delta_format = portfolio_cfg["time_delta_format"]

        if init_cash is None:
            init_cash = portfolio_cfg["init_cash"]
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if checks.is_int(init_cash) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg["init_position"]
        if init_price is None:
            init_price = portfolio_cfg["init_price"]
        if cash_deposits is None:
            cash_deposits = portfolio_cfg["cash_deposits"]
        if cash_earnings is None:
            cash_earnings = portfolio_cfg["cash_earnings"]
        if cash_dividends is None:
            cash_dividends = portfolio_cfg["cash_dividends"]
        if cash_sharing is None:
            cash_sharing = portfolio_cfg["cash_sharing"]
        if cash_sharing and group_by is None:
            group_by = True
        if from_ago is None:
            from_ago = portfolio_cfg["from_ago"]
        if call_seq is None:
            call_seq = portfolio_cfg["call_seq"]
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if checks.is_int(call_seq):
            if call_seq == CallSeqType.Auto:
                auto_call_seq = True
                call_seq = None
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg["attach_call_seq"]
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg["ffill_val_price"]
        if update_value is None:
            update_value = portfolio_cfg["update_value"]
        if fill_returns is None:
            fill_returns = portfolio_cfg["fill_returns"]
        if seed is None:
            seed = portfolio_cfg["seed"]
        if seed is not None:
            set_seed(seed)
        if flexible_mode:
            if fill_returns:
                raise ValueError("Argument fill_returns cannot be used in flexible mode")
            if in_outputs is not None and not checks.is_namedtuple(in_outputs):
                in_outputs = to_mapping(in_outputs)
                in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        else:
            if in_outputs is not None:
                raise ValueError("Argument in_outputs cannot be used in fixed mode")
        if group_by is None:
            group_by = portfolio_cfg["group_by"]
        if freq is None:
            freq = portfolio_cfg["freq"]
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_kwargs = merge_dicts(portfolio_cfg["broadcast_kwargs"], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get("require_kwargs", {})
        template_context = merge_dicts(portfolio_cfg["template_context"], template_context)
        if bm_close is None:
            bm_close = portfolio_cfg["bm_close"]

        # Prepare the simulation
        broadcastable_args = dict(
            cash_earnings=cash_earnings,
            cash_dividends=cash_dividends,
            size=size,
            price=price,
            size_type=size_type,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            reject_prob=reject_prob,
            price_area_vio_mode=price_area_vio_mode,
            lock_cash=lock_cash,
            allow_partial=allow_partial,
            raise_reject=raise_reject,
            log=log,
            val_price=val_price,
            accumulate=accumulate,
            upon_long_conflict=upon_long_conflict,
            upon_short_conflict=upon_short_conflict,
            upon_dir_conflict=upon_dir_conflict,
            upon_opposite_entry=upon_opposite_entry,
            order_type=order_type,
            limit_delta=limit_delta,
            limit_tif=limit_tif,
            limit_expiry=limit_expiry,
            limit_reverse=limit_reverse,
            upon_adj_limit_conflict=upon_adj_limit_conflict,
            upon_opp_limit_conflict=upon_opp_limit_conflict,
            sl_stop=sl_stop,
            tsl_stop=tsl_stop,
            tsl_th=tsl_th,
            tp_stop=tp_stop,
            stop_entry_price=stop_entry_price,
            stop_exit_price=stop_exit_price,
            stop_exit_type=stop_exit_type,
            stop_order_type=stop_order_type,
            stop_limit_delta=stop_limit_delta,
            upon_stop_update=upon_stop_update,
            upon_adj_stop_conflict=upon_adj_stop_conflict,
            upon_opp_stop_conflict=upon_opp_stop_conflict,
            delta_format=delta_format,
            time_delta_format=time_delta_format,
            open=open,
            high=high,
            low=low,
            close=close,
            from_ago=from_ago,
        )
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args["bm_close"] = bm_close
        else:
            broadcastable_args["bm_close"] = None
        if not signal_func_mode:
            if ls_mode:
                broadcastable_args["entries"] = entries
                broadcastable_args["exits"] = exits
                broadcastable_args["short_entries"] = short_entries
                broadcastable_args["short_exits"] = short_exits
            else:
                broadcastable_args["entries"] = entries
                broadcastable_args["exits"] = exits
                broadcastable_args["direction"] = direction
        broadcastable_args = {**broadcastable_args, **broadcast_named_args}
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=True,
                reindex_kwargs=dict(
                    cash_earnings=dict(fill_value=0.0),
                    cash_dividends=dict(fill_value=0.0),
                    entries=dict(fill_value=False),
                    exits=dict(fill_value=False),
                    short_entries=dict(fill_value=False),
                    short_exits=dict(fill_value=False),
                    size=dict(fill_value=np.nan),
                    price=dict(fill_value=np.nan),
                    size_type=dict(fill_value=SizeType.Amount),
                    direction=dict(fill_value=Direction.Both),
                    fees=dict(fill_value=0.0),
                    fixed_fees=dict(fill_value=0.0),
                    slippage=dict(fill_value=0.0),
                    min_size=dict(fill_value=np.nan),
                    max_size=dict(fill_value=np.nan),
                    size_granularity=dict(fill_value=np.nan),
                    reject_prob=dict(fill_value=0.0),
                    price_area_vio_mode=dict(fill_value=PriceAreaVioMode.Ignore),
                    lock_cash=dict(fill_value=False),
                    allow_partial=dict(fill_value=True),
                    raise_reject=dict(fill_value=False),
                    log=dict(fill_value=False),
                    val_price=dict(fill_value=np.nan),
                    accumulate=dict(fill_value=False),
                    upon_long_conflict=dict(fill_value=ConflictMode.Ignore),
                    upon_short_conflict=dict(fill_value=ConflictMode.Ignore),
                    upon_dir_conflict=dict(fill_value=DirectionConflictMode.Ignore),
                    upon_opposite_entry=dict(fill_value=OppositeEntryMode.ReverseReduce),
                    order_type=dict(fill_value=OrderType.Market),
                    limit_delta=dict(fill_value=np.nan),
                    limit_tif=dict(fill_value=-1),
                    limit_expiry=dict(fill_value=-1),
                    limit_reverse=dict(fill_value=False),
                    upon_adj_limit_conflict=dict(fill_value=PendingConflictMode.KeepIgnore),
                    upon_opp_limit_conflict=dict(fill_value=PendingConflictMode.CancelExecute),
                    sl_stop=dict(fill_value=np.nan),
                    tsl_stop=dict(fill_value=np.nan),
                    tsl_th=dict(fill_value=np.nan),
                    tp_stop=dict(fill_value=np.nan),
                    stop_entry_price=dict(fill_value=StopEntryPrice.Close),
                    stop_exit_price=dict(fill_value=StopExitPrice.Stop),
                    stop_exit_type=dict(fill_value=StopExitType.Close),
                    stop_order_type=dict(fill_value=OrderType.Market),
                    stop_limit_delta=dict(fill_value=np.nan),
                    upon_stop_update=dict(fill_value=StopUpdateMode.Override),
                    upon_adj_stop_conflict=dict(fill_value=PendingConflictMode.KeepExecute),
                    upon_opp_stop_conflict=dict(fill_value=PendingConflictMode.KeepExecute),
                    delta_format=dict(fill_value=DeltaFormat.Percent),
                    time_delta_format=dict(fill_value=TimeDeltaFormat.Index),
                    open=dict(fill_value=np.nan),
                    high=dict(fill_value=np.nan),
                    low=dict(fill_value=np.nan),
                    close=dict(fill_value=np.nan),
                    bm_close=dict(fill_value=np.nan),
                    from_ago=dict(fill_value=0),
                ),
                wrapper_kwargs=dict(
                    freq=freq,
                    group_by=group_by,
                ),
            ),
            broadcast_kwargs,
        )
        broadcasted_args, wrapper = broadcast(broadcastable_args, return_wrapper=True, **broadcast_kwargs)
        if not wrapper.group_select and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        cash_earnings = broadcasted_args.pop("cash_earnings")
        cash_dividends = broadcasted_args.pop("cash_dividends")
        target_shape_2d = wrapper.shape_2d
        index = wrapper.ns_index
        freq = wrapper.ns_freq

        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        init_price = np.require(np.broadcast_to(init_price, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            cash_deposits,
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            keep_flex=True,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=require_kwargs,
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if call_seq is None and attach_call_seq:
            call_seq = CallSeqType.Default
        if call_seq is not None:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
            else:
                call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if max_logs is None:
            _log = broadcasted_args["log"]
            if _log.size == 1:
                max_logs = target_shape_2d[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and target_shape_2d[0] > 1:
                    max_logs = target_shape_2d[0] * int(np.any(_log))
                else:
                    max_logs = int(np.max(np.sum(_log, axis=0)))
        if use_stops is None:
            if flexible_mode:
                use_stops = True
            else:
                if (
                    not np.any(broadcasted_args["sl_stop"])
                    and not np.any(broadcasted_args["tsl_stop"])
                    and not np.any(broadcasted_args["tp_stop"])
                ):
                    use_stops = False
                else:
                    use_stops = True

        # Convert strings to numbers
        if "direction" in broadcasted_args:
            broadcasted_args["direction"] = map_enum_fields(broadcasted_args["direction"], Direction)
        broadcasted_args["price"] = map_enum_fields(broadcasted_args["price"], PriceType, ignore_type=(int, float))
        broadcasted_args["size_type"] = map_enum_fields(broadcasted_args["size_type"], SizeType)
        broadcasted_args["price_area_vio_mode"] = map_enum_fields(
            broadcasted_args["price_area_vio_mode"],
            PriceAreaVioMode,
        )
        broadcasted_args["val_price"] = map_enum_fields(
            broadcasted_args["val_price"],
            ValPriceType,
            ignore_type=(int, float),
        )
        broadcasted_args["accumulate"] = map_enum_fields(
            broadcasted_args["accumulate"],
            AccumulationMode,
            ignore_type=(int, bool),
        )
        broadcasted_args["upon_long_conflict"] = map_enum_fields(broadcasted_args["upon_long_conflict"], ConflictMode)
        broadcasted_args["upon_short_conflict"] = map_enum_fields(broadcasted_args["upon_short_conflict"], ConflictMode)
        broadcasted_args["upon_dir_conflict"] = map_enum_fields(
            broadcasted_args["upon_dir_conflict"],
            DirectionConflictMode,
        )
        broadcasted_args["upon_opposite_entry"] = map_enum_fields(
            broadcasted_args["upon_opposite_entry"],
            OppositeEntryMode,
        )
        broadcasted_args["order_type"] = map_enum_fields(broadcasted_args["order_type"], OrderType)
        limit_tif = broadcasted_args["limit_tif"]
        if limit_tif.dtype == object:
            if limit_tif.ndim in (0, 1):
                limit_tif = pd.to_timedelta(limit_tif)
                if isinstance(limit_tif, pd.Timedelta):
                    limit_tif = limit_tif.to_timedelta64()
                else:
                    limit_tif = limit_tif.values
            else:
                limit_tif_cols = []
                for col in range(limit_tif.shape[1]):
                    limit_tif_col = pd.to_timedelta(limit_tif[:, col])
                    limit_tif_cols.append(limit_tif_col.values)
                limit_tif = np.column_stack(limit_tif_cols)
        broadcasted_args["limit_tif"] = limit_tif.astype(np.int64)
        limit_expiry = broadcasted_args["limit_expiry"]
        if limit_expiry.dtype == object:
            if limit_expiry.ndim in (0, 1):
                limit_expiry = pd.to_datetime(limit_expiry).tz_localize(None)
                if isinstance(limit_expiry, pd.Timestamp):
                    limit_expiry = limit_expiry.to_datetime64()
                else:
                    limit_expiry = limit_expiry.values
            else:
                limit_expiry_cols = []
                for col in range(limit_expiry.shape[1]):
                    limit_expiry_col = pd.to_datetime(limit_expiry[:, col]).tz_localize(None)
                    limit_expiry_cols.append(limit_expiry_col.values)
                limit_expiry = np.column_stack(limit_expiry_cols)
        broadcasted_args["limit_expiry"] = limit_expiry.astype(np.int64)
        broadcasted_args["upon_adj_limit_conflict"] = map_enum_fields(
            broadcasted_args["upon_adj_limit_conflict"],
            PendingConflictMode,
        )
        broadcasted_args["upon_opp_limit_conflict"] = map_enum_fields(
            broadcasted_args["upon_opp_limit_conflict"],
            PendingConflictMode,
        )
        broadcasted_args["stop_entry_price"] = map_enum_fields(
            broadcasted_args["stop_entry_price"],
            StopEntryPrice,
            ignore_type=(int, float),
        )
        broadcasted_args["stop_exit_price"] = map_enum_fields(
            broadcasted_args["stop_exit_price"],
            StopExitPrice,
            ignore_type=(int, float),
        )
        broadcasted_args["stop_exit_type"] = map_enum_fields(broadcasted_args["stop_exit_type"], StopExitType)
        broadcasted_args["stop_order_type"] = map_enum_fields(broadcasted_args["stop_order_type"], OrderType)
        broadcasted_args["upon_stop_update"] = map_enum_fields(broadcasted_args["upon_stop_update"], StopUpdateMode)
        broadcasted_args["upon_adj_stop_conflict"] = map_enum_fields(
            broadcasted_args["upon_adj_stop_conflict"],
            PendingConflictMode,
        )
        broadcasted_args["upon_opp_stop_conflict"] = map_enum_fields(
            broadcasted_args["upon_opp_stop_conflict"],
            PendingConflictMode,
        )
        broadcasted_args["delta_format"] = map_enum_fields(broadcasted_args["delta_format"], DeltaFormat)
        broadcasted_args["time_delta_format"] = map_enum_fields(broadcasted_args["time_delta_format"], TimeDeltaFormat)

        # Check data types
        if "entries" in broadcasted_args:
            checks.assert_subdtype(broadcasted_args["entries"], np.bool_)
        if "exits" in broadcasted_args:
            checks.assert_subdtype(broadcasted_args["exits"], np.bool_)
        if "short_entries" in broadcasted_args:
            checks.assert_subdtype(broadcasted_args["short_entries"], np.bool_)
        if "short_exits" in broadcasted_args:
            checks.assert_subdtype(broadcasted_args["short_exits"], np.bool_)
        if "direction" in broadcasted_args:
            checks.assert_subdtype(broadcasted_args["direction"], np.integer)
        checks.assert_subdtype(broadcasted_args["size"], np.number)
        checks.assert_subdtype(broadcasted_args["price"], np.number)
        checks.assert_subdtype(broadcasted_args["size_type"], np.integer)
        checks.assert_subdtype(broadcasted_args["fees"], np.number)
        checks.assert_subdtype(broadcasted_args["fixed_fees"], np.number)
        checks.assert_subdtype(broadcasted_args["slippage"], np.number)
        checks.assert_subdtype(broadcasted_args["min_size"], np.number)
        checks.assert_subdtype(broadcasted_args["max_size"], np.number)
        checks.assert_subdtype(broadcasted_args["size_granularity"], np.number)
        checks.assert_subdtype(broadcasted_args["reject_prob"], np.number)
        checks.assert_subdtype(broadcasted_args["price_area_vio_mode"], np.integer)
        checks.assert_subdtype(broadcasted_args["lock_cash"], np.bool_)
        checks.assert_subdtype(broadcasted_args["allow_partial"], np.bool_)
        checks.assert_subdtype(broadcasted_args["raise_reject"], np.bool_)
        checks.assert_subdtype(broadcasted_args["log"], np.bool_)
        checks.assert_subdtype(broadcasted_args["val_price"], np.number)
        checks.assert_subdtype(broadcasted_args["accumulate"], (np.integer, np.bool_))
        checks.assert_subdtype(broadcasted_args["upon_long_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_short_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_dir_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_opposite_entry"], np.integer)
        checks.assert_subdtype(broadcasted_args["order_type"], np.integer)
        checks.assert_subdtype(broadcasted_args["limit_delta"], np.number)
        checks.assert_subdtype(broadcasted_args["limit_tif"], np.integer)
        checks.assert_subdtype(broadcasted_args["limit_expiry"], np.integer)
        checks.assert_subdtype(broadcasted_args["limit_reverse"], np.bool_)
        checks.assert_subdtype(broadcasted_args["upon_adj_limit_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_opp_limit_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["sl_stop"], np.number)
        checks.assert_subdtype(broadcasted_args["tsl_stop"], np.number)
        checks.assert_subdtype(broadcasted_args["tsl_th"], np.number)
        checks.assert_subdtype(broadcasted_args["tp_stop"], np.number)
        checks.assert_subdtype(broadcasted_args["stop_entry_price"], np.number)
        checks.assert_subdtype(broadcasted_args["stop_exit_price"], np.number)
        checks.assert_subdtype(broadcasted_args["stop_exit_type"], np.integer)
        checks.assert_subdtype(broadcasted_args["stop_order_type"], np.integer)
        checks.assert_subdtype(broadcasted_args["stop_limit_delta"], np.number)
        checks.assert_subdtype(broadcasted_args["upon_stop_update"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_adj_stop_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["upon_opp_stop_conflict"], np.integer)
        checks.assert_subdtype(broadcasted_args["delta_format"], np.integer)
        checks.assert_subdtype(broadcasted_args["time_delta_format"], np.integer)
        checks.assert_subdtype(broadcasted_args["open"], np.number)
        checks.assert_subdtype(broadcasted_args["high"], np.number)
        checks.assert_subdtype(broadcasted_args["low"], np.number)
        checks.assert_subdtype(broadcasted_args["close"], np.number)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args["bm_close"], np.number)
        checks.assert_subdtype(cs_group_lens, np.integer)
        checks.assert_subdtype(broadcasted_args["from_ago"], np.integer)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(init_price, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(cash_dividends, np.number)

        # Prepare arguments
        template_context = merge_dicts(
            broadcasted_args,
            dict(
                target_shape=target_shape_2d,
                index=index,
                freq=freq,
                group_lens=group_lens if flexible_mode else cs_group_lens,
                cs_group_lens=cs_group_lens,
                call_seq=call_seq,
                init_cash=init_cash,
                init_position=init_position,
                init_price=init_price,
                cash_deposits=cash_deposits,
                cash_earnings=cash_earnings,
                cash_dividends=cash_dividends,
                use_stops=use_stops,
                adjust_func_nb=adjust_func_nb,
                adjust_args=adjust_args,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                fill_returns=fill_returns,
                max_orders=max_orders,
                max_logs=max_logs,
                in_outputs=in_outputs,
                wrapper=wrapper,
            ),
            template_context,
        )
        if flexible_mode:
            in_outputs = deep_substitute(in_outputs, template_context, sub_id="in_outputs")
            post_segment_args = deep_substitute(post_segment_args, template_context, sub_id="post_segment_args")
            if signal_func_mode:
                signal_args = deep_substitute(signal_args, template_context, sub_id="signal_args")
            else:
                adjust_args = deep_substitute(adjust_args, template_context, sub_id="adjust_args")
                if ls_mode:
                    signal_args = (
                        broadcasted_args.pop("entries"),
                        broadcasted_args.pop("exits"),
                        broadcasted_args.pop("short_entries"),
                        broadcasted_args.pop("short_exits"),
                        broadcasted_args["from_ago"],
                        adjust_func_nb,
                        adjust_args,
                    )
                    chunked = ch.specialize_chunked_option(
                        chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                None,
                                ArgsTaker(),
                            )
                        ),
                    )
                else:
                    signal_args = (
                        broadcasted_args.pop("entries"),
                        broadcasted_args.pop("exits"),
                        broadcasted_args.pop("direction"),
                        broadcasted_args["from_ago"],
                        adjust_func_nb,
                        adjust_args,
                    )
                    chunked = ch.specialize_chunked_option(
                        chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                None,
                                ArgsTaker(),
                            )
                        ),
                    )
            for k in broadcast_named_args:
                broadcasted_args.pop(k)
            bm_close = broadcasted_args.pop("bm_close", None)

            # Perform the simulation
            func = jit_reg.resolve_option(nb.simulate_from_signal_func_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            sim_out = func(
                target_shape=target_shape_2d,
                group_lens=group_lens,
                cash_sharing=cash_sharing,
                index=index,
                freq=freq,
                init_cash=init_cash,
                init_position=init_position,
                init_price=init_price,
                cash_deposits=cash_deposits,
                cash_earnings=cash_earnings,
                cash_dividends=cash_dividends,
                signal_func_nb=signal_func_nb,
                signal_args=signal_args,
                post_segment_func_nb=post_segment_func_nb,
                post_segment_args=post_segment_args,
                use_stops=use_stops,
                call_seq=call_seq,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                max_orders=max_orders,
                max_logs=max_logs,
                in_outputs=in_outputs,
                **broadcasted_args,
            )
        else:
            entries = broadcasted_args.pop("entries")
            exits = broadcasted_args.pop("exits")
            if ls_mode:
                long_entries = entries
                long_exits = exits
                short_entries = broadcasted_args.pop("short_entries")
                short_exits = broadcasted_args.pop("short_exits")
            else:
                direction = broadcasted_args.pop("direction")
                if direction.size == 1:
                    _direction = direction.item(0)
                    if _direction == Direction.LongOnly:
                        long_entries = entries
                        long_exits = exits
                        short_entries = np.array([[False]])
                        short_exits = np.array([[False]])
                    elif _direction == Direction.ShortOnly:
                        long_entries = np.array([[False]])
                        long_exits = np.array([[False]])
                        short_entries = entries
                        short_exits = exits
                    else:
                        long_entries = entries
                        long_exits = np.array([[False]])
                        short_entries = exits
                        short_exits = np.array([[False]])
                else:
                    long_entries, long_exits, short_entries, short_exits = nb.dir_to_ls_signals_nb(
                        target_shape=target_shape_2d,
                        entries=entries,
                        exits=exits,
                        direction=direction,
                    )

            for k in broadcast_named_args:
                broadcasted_args.pop(k)
            bm_close = broadcasted_args.pop("bm_close", None)

            # Perform the simulation
            func = jit_reg.resolve_option(nb.simulate_from_signals_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            sim_out = func(
                target_shape=target_shape_2d,
                group_lens=cs_group_lens,  # group only if cash sharing is enabled to speed up
                index=index,
                freq=freq,
                init_cash=init_cash,
                init_position=init_position,
                init_price=init_price,
                cash_deposits=cash_deposits,
                cash_earnings=cash_earnings,
                cash_dividends=cash_dividends,
                long_entries=long_entries,
                long_exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                use_stops=use_stops,
                call_seq=call_seq,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                fill_returns=fill_returns,
                max_orders=max_orders,
                max_logs=max_logs,
                **broadcasted_args,
            )

        # Create an instance
        if "orders_cls" not in kwargs:
            kwargs["orders_cls"] = FSOrders
        return cls(
            wrapper,
            broadcasted_args["close"],
            sim_out.order_records,
            open=broadcasted_args["open"] if not open_none else None,
            high=broadcasted_args["high"] if not high_none else None,
            low=broadcasted_args["low"] if not low_none else None,
            log_records=sim_out.log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=sim_out.cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs,
        )

    @classmethod
    def from_holding(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        direction: tp.Optional[int] = None,
        at_first_valid_in: tp.Optional[str] = "close",
        close_at_end: tp.Optional[bool] = None,
        dynamic_mode: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Simulate portfolio from plain holding using signals.

        If `close_at_end` is True, will place an opposite signal at the very end.

        `**kwargs` are passed to the class method `Portfolio.from_signals`.

        Usage:
            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_holding(close)
            >>> pf.final_value
            500.0
            ```
        """
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if direction is None:
            direction = portfolio_cfg["hold_direction"]
        direction = map_enum_fields(direction, Direction)
        if not checks.is_int(direction):
            raise TypeError("Direction must be a scalar")
        if close_at_end is None:
            close_at_end = portfolio_cfg["close_at_end"]

        if dynamic_mode:
            return cls.from_signals(
                close,
                signal_func_nb=nb.holding_enex_signal_func_nb,
                signal_args=(direction, close_at_end),
                accumulate=False,
                **kwargs,
            )

        def _entries(wrapper, new_objs):
            if at_first_valid_in is None:
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            ts = new_objs[at_first_valid_in]
            valid_index = generic_nb.first_valid_index_nb(ts)
            if (valid_index == -1).all():
                return np.array([[False]])
            if (valid_index == 0).all():
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            entries = np.full(wrapper.shape_2d, False)
            entries[valid_index, np.arange(wrapper.shape_2d[1])] = True
            return entries

        def _exits(wrapper):
            if close_at_end:
                exits = np.full((wrapper.shape_2d[0], 1), False)
                exits[-1] = True
            else:
                exits = np.array([[False]])
            return exits

        return cls.from_signals(
            close,
            entries=RepFunc(_entries),
            exits=RepFunc(_exits),
            direction=direction,
            accumulate=False,
            **kwargs,
        )

    @classmethod
    def from_random_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        n: tp.Optional[tp.ArrayLike] = None,
        prob: tp.Optional[tp.ArrayLike] = None,
        entry_prob: tp.Optional[tp.ArrayLike] = None,
        exit_prob: tp.Optional[tp.ArrayLike] = None,
        param_product: bool = False,
        seed: tp.Optional[int] = None,
        run_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Simulate portfolio from random entry and exit signals.

        Generates signals based either on the number of signals `n` or the probability
        of encountering a signal `prob`.

        * If `n` is set, see `vectorbtpro.signals.generators.RANDNX`.
        * If `prob` is set, see `vectorbtpro.signals.generators.RPROBNX`.

        Based on `Portfolio.from_signals`.

        !!! note
            To generate random signals, the shape of `close` is used. Broadcasting with other
            arrays happens after the generation.

        Usage:
            * Test multiple combinations of random entries and exits:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_random_signals(close, n=[2, 1, 0], seed=42)
            >>> pf.orders.count()
            randnx_n
            2    4
            1    2
            0    0
            Name: count, dtype: int64
            ```

            * Test the Cartesian product of entry and exit encounter probabilities:

            ```pycon
            >>> pf = vbt.Portfolio.from_random_signals(
            ...     close,
            ...     entry_prob=[0, 0.5, 1],
            ...     exit_prob=[0, 0.5, 1],
            ...     param_product=True,
            ...     seed=42)
            >>> pf.orders.count()
            rprobnx_entry_prob  rprobnx_exit_prob
            0.0                 0.0                  0
                                0.5                  0
                                1.0                  0
            0.5                 0.0                  1
                                0.5                  4
                                1.0                  3
            1.0                 0.0                  1
                                0.5                  4
                                1.0                  5
            Name: count, dtype: int64
            ```
        """
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if isinstance(close, Data):
            data = close
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            close_wrapper = data.symbol_wrapper
        else:
            close = to_pd_array(close)
            close_wrapper = ArrayWrapper.from_obj(close)
            data = close
        if entry_prob is None:
            entry_prob = prob
        if exit_prob is None:
            exit_prob = prob
        if seed is None:
            seed = portfolio_cfg["seed"]
        if run_kwargs is None:
            run_kwargs = {}

        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or entry_prob and exit_prob must be provided")
        if n is not None:
            rand = RANDNX.run(
                n=n,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rand.entries
            exits = rand.exits
        elif entry_prob is not None and exit_prob is not None:
            rprobnx = RPROBNX.run(
                entry_prob=entry_prob,
                exit_prob=exit_prob,
                param_product=param_product,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rprobnx.entries
            exits = rprobnx.exits
        else:
            raise ValueError("At least n or entry_prob and exit_prob must be provided")

        return cls.from_signals(data, entries, exits, seed=seed, **kwargs)

    @classmethod
    def from_optimizer(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        optimizer: PortfolioOptimizer,
        squeeze_groups: bool = True,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        size_type: tp.ArrayLike = "targetpercent",
        direction: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = True,
        call_seq: tp.Optional[tp.ArrayLike] = "auto",
        group_by: tp.GroupByLike = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Build portfolio from an optimizer of type `vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer`.

        Uses `Portfolio.from_orders` as the base simulation method.

        The size type is 'targetpercent'. If there are positive and negative values, the direction
        is automatically set to 'both', otherwise to 'longonly' for positive-only and `shortonly`
        for negative-only values. Also, the cash sharing is set to True, the call sequence is set
        to 'auto', and the grouper is set to the grouper of the optimizer by default.

        Usage:
            ```pycon
            >>> close = pd.DataFrame({
            ...     "MSFT": [1, 2, 3, 4, 5],
            ...     "GOOG": [5, 4, 3, 2, 1],
            ...     "AAPL": [1, 2, 3, 2, 1]
            ... }, index=pd.date_range(start="2020-01-01", periods=5))

            >>> pf_opt = vbt.PortfolioOptimizer.from_random(
            ...     close.vbt.wrapper,
            ...     every="2D",
            ...     seed=42
            ... )
            >>> pf_opt.fill_allocations()
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812
            2020-01-02        NaN       NaN       NaN
            2020-01-03   0.657381  0.171323  0.171296
            2020-01-04        NaN       NaN       NaN
            2020-01-05   0.038078  0.567845  0.394077

            >>> pf = vbt.Portfolio.from_optimizer(close, pf_opt)
            >>> pf.get_asset_value(group_by=False).vbt / pf.value
            alloc_group                         group
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812  << rebalanced
            2020-01-02   0.251907  0.255771  0.492322
            2020-01-03   0.657381  0.171323  0.171296  << rebalanced
            2020-01-04   0.793277  0.103369  0.103353
            2020-01-05   0.038078  0.567845  0.394077  << rebalanced
            ```
        """
        size = optimizer.fill_allocations(squeeze_groups=squeeze_groups, dropna=dropna, fill_value=fill_value)
        if direction is None:
            pos_size_any = (size.values > 0).any()
            neg_size_any = (size.values < 0).any()
            if pos_size_any and neg_size_any:
                direction = "both"
            elif pos_size_any:
                direction = "longonly"
            else:
                direction = "shortonly"
                size = size.abs()
        if group_by is None:

            def _substitute_group_by(index):
                columns = optimizer.wrapper.columns
                if squeeze_groups and optimizer.wrapper.grouped_ndim == 1:
                    columns = columns.droplevel(level=0)
                if not index.equals(columns):
                    if "symbol" in index.names:
                        return ExceptLevel("symbol")
                    raise ValueError("Column hierarchy has changed. Disable squeeze_groups and provide group_by.")
                return optimizer.wrapper.grouper.group_by

            group_by = RepFunc(_substitute_group_by)
        return cls.from_orders(
            close,
            size=size,
            size_type=size_type,
            direction=direction,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            group_by=group_by,
            **kwargs,
        )

    @classmethod
    def from_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        order_func_nb: tp.Union[nb.OrderFuncT, nb.FlexOrderFuncT],
        *order_args,
        flexible: tp.Optional[bool] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        segment_mask: tp.Optional[tp.ArrayLike] = None,
        call_pre_segment: tp.Optional[bool] = None,
        call_post_segment: tp.Optional[bool] = None,
        pre_sim_func_nb: nb.PreSimFuncT = nb.no_pre_func_nb,
        pre_sim_args: tp.Args = (),
        post_sim_func_nb: nb.PostSimFuncT = nb.no_post_func_nb,
        post_sim_args: tp.Args = (),
        pre_group_func_nb: nb.PreGroupFuncT = nb.no_pre_func_nb,
        pre_group_args: tp.Args = (),
        post_group_func_nb: nb.PostGroupFuncT = nb.no_post_func_nb,
        post_group_args: tp.Args = (),
        pre_row_func_nb: nb.PreRowFuncT = nb.no_pre_func_nb,
        pre_row_args: tp.Args = (),
        post_row_func_nb: nb.PostRowFuncT = nb.no_post_func_nb,
        post_row_args: tp.Args = (),
        pre_segment_func_nb: nb.PreSegmentFuncT = nb.no_pre_func_nb,
        pre_segment_args: tp.Args = (),
        post_segment_func_nb: nb.PostSegmentFuncT = nb.no_post_func_nb,
        post_segment_args: tp.Args = (),
        post_order_func_nb: nb.PostOrderFuncT = nb.no_post_func_nb,
        post_order_args: tp.Args = (),
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_pos_record: tp.Optional[bool] = None,
        track_value: tp.Optional[bool] = None,
        row_wise: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        keep_inout_raw: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> PortfolioT:
        """Build portfolio from a custom order function.

        !!! hint
            See `vectorbtpro.portfolio.nb.from_order_func.simulate_nb` for illustrations and argument definitions.

        For more details on individual simulation functions:

        * not `row_wise` and not `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.simulate_nb`
        * not `row_wise` and `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.flex_simulate_nb`
        * `row_wise` and not `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.simulate_row_wise_nb`
        * `row_wise` and `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.flex_simulate_row_wise_nb`

        Args:
            close (array_like or Data): Latest asset price at each time step.
                Will broadcast.

                If an instance of `vectorbtpro.data.base.Data`, will extract the open, high,
                low, and close price.

                Used for calculating unrealized PnL and portfolio value.
            order_func_nb (callable): Order generation function.
            *order_args: Arguments passed to `order_func_nb`.
            flexible (bool): Whether to simulate using a flexible order function.

                This lifts the limit of one order per tick and symbol.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            init_price (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): Whether to share cash within the same group.

                If `group_by` is None, `group_by` becomes True to form a single group with cash sharing.
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                * Use `vectorbtpro.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

                !!! note
                    CallSeqType.Auto must be implemented manually.
                    Use `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_1d_nb`
                    or `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_out_1d_nb` in `pre_segment_func_nb`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            segment_mask (int or array_like of bool): Mask of whether a particular segment should be executed.

                Supplying an integer will activate every n-th row.
                Supplying a boolean or an array of boolean will broadcast to the number of rows and groups.

                Does not broadcast together with `close` and `broadcast_named_args`, only against the final shape.
            call_pre_segment (bool): Whether to call `pre_segment_func_nb` regardless of `segment_mask`.
            call_post_segment (bool): Whether to call `post_segment_func_nb` regardless of `segment_mask`.
            pre_sim_func_nb (callable): Function called before simulation.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_sim_args (tuple): Packed arguments passed to `pre_sim_func_nb`.
                Defaults to `()`.
            post_sim_func_nb (callable): Function called after simulation.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_sim_args (tuple): Packed arguments passed to `post_sim_func_nb`.
                Defaults to `()`.
            pre_group_func_nb (callable): Function called before each group.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.

                Called only if `row_wise` is False.
            pre_group_args (tuple): Packed arguments passed to `pre_group_func_nb`.
                Defaults to `()`.
            post_group_func_nb (callable): Function called after each group.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.

                Called only if `row_wise` is False.
            post_group_args (tuple): Packed arguments passed to `post_group_func_nb`.
                Defaults to `()`.
            pre_row_func_nb (callable): Function called before each row.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.

                Called only if `row_wise` is True.
            pre_row_args (tuple): Packed arguments passed to `pre_row_func_nb`.
                Defaults to `()`.
            post_row_func_nb (callable): Function called after each row.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.

                Called only if `row_wise` is True.
            post_row_args (tuple): Packed arguments passed to `post_row_func_nb`.
                Defaults to `()`.
            pre_segment_func_nb (callable): Function called before each segment.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_segment_args (tuple): Packed arguments passed to `pre_segment_func_nb`.
                Defaults to `()`.
            post_segment_func_nb (callable): Function called after each segment.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
                Defaults to `()`.
            post_order_func_nb (callable): Callback that is called after the order has been processed.
            post_order_args (tuple): Packed arguments passed to `post_order_func_nb`.
                Defaults to `()`.
            open (array_like of float): See `Portfolio.from_orders`.
            high (array_like of float): See `Portfolio.from_orders`.
            low (array_like of float): See `Portfolio.from_orders`.
            ffill_val_price (bool): Whether to track valuation price only if it's known.

                Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp.
            update_value (bool): Whether to update group value after each filled order.
            fill_pos_record (bool): Whether to fill position record.

                Disable this to make simulation faster for simple use cases.
            track_value (bool): Whether to track value metrics such as
                the current valuation price, value, and return.

                Disable this to make simulation faster for simple use cases.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.
            max_orders (int): The max number of order records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape.

                Set to a lower number if you run out of memory, to 0 to not fill, and to a higher number
                if there are more than one order expected at each timestamp.
            max_logs (int): The max number of log records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape.

                Set to a lower number if you run out of memory, to 0 to not fill, and to a higher number
                if there are more than one order expected at each timestamp.
            in_outputs (mapping_like): Mapping with in-output objects.

                Will be available via `Portfolio.in_outputs` as a named tuple.

                To substitute `Portfolio` attributes, provide already broadcasted and grouped objects,
                for example, by using `broadcast_named_args` and templates. Also see
                `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.

                When chunking, make sure to provide the chunk taking specification and the merging function.
                See `vectorbtpro.portfolio.chunking.merge_sim_outs`.

                !!! note
                    When using Numba below 0.54, `in_outputs` cannot be a mapping, but must be a named tuple
                    defined globally so Numba can introspect its attributes for pickling.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): See `Portfolio.from_signals`.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_context (mapping): See `Portfolio.from_signals`.
            keep_inout_raw (bool): Whether to keep arrays that can be edited in-place raw when broadcasting.

                Disable this to be able to edit `segment_mask`, `cash_deposits`, and
                `cash_earnings` during the simulation.
            jitted (any): See `Portfolio.from_orders`.

                !!! note
                    Disabling jitting will not disable jitter (such as Numba) on other functions,
                    only on the main (simulation) function. If neccessary, you should ensure that every other
                    function is not compiled as well. For example, when working with Numba, you can do this
                    by using the `py_func` attribute of that function. Or, you can disable Numba
                    entirely by running `os.environ['NUMBA_DISABLE_JIT'] = '1'` before importing vectorbtpro.

                !!! warning
                    Parallelization assumes that groups are independent and there is no data flowing between them.
            chunked (any): See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        !!! note
            All passed functions must be Numba-compiled if Numba is enabled.

            Also see notes on `Portfolio.from_orders`.

        !!! note
            In contrast to other methods, the valuation price is previous `close` instead of the order price
            since the price of an order is unknown before the call (which is more realistic by the way).
            You can still override the valuation price in `pre_segment_func_nb`.

        Usage:
            * Buy 10 units each tick using closing price:

            ```pycon
            >>> @njit
            ... def order_func_nb(c, size):
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_order_func(close, order_func_nb, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Reverse each position by first closing it. Keep state of last position to determine
            which position to open next (just as an example, there are easier ways to do this):

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     last_pos_state = np.array([-1])
            ...     return (last_pos_state,)

            >>> @njit
            ... def order_func_nb(c, last_pos_state):
            ...     if c.position_now != 0:
            ...         return vbt.pf_nb.close_position_nb()
            ...
            ...     if last_pos_state[0] == 1:
            ...         size = -np.inf  # open short
            ...         last_pos_state[0] = -1
            ...     else:
            ...         size = np.inf  # open long
            ...         last_pos_state[0] = 1
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb,
            ...     pre_group_func_nb=pre_group_func_nb
            ... )

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * Equal-weighted portfolio as in the example under `vectorbtpro.portfolio.nb.from_order_func.simulate_nb`:

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     order_value_out = np.empty(c.group_len, dtype=np.float_)
            ...     return (order_value_out,)

            >>> @njit
            ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.last_val_price[col] = vbt.pf_nb.select_from_col_nb(c, col, price)
            ...     vbt.pf_nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            ...     return ()

            >>> @njit
            ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
            ...     return vbt.pf_nb.order_nb(
            ...         size=vbt.pf_nb.select_nb(c, size),
            ...         price=vbt.pf_nb.select_nb(c, price),
            ...         size_type=vbt.pf_nb.select_nb(c, size_type),
            ...         direction=vbt.pf_nb.select_nb(c, direction),
            ...         fees=vbt.pf_nb.select_nb(c, fees),
            ...         fixed_fees=vbt.pf_nb.select_nb(c, fixed_fees),
            ...         slippage=vbt.pf_nb.select_nb(c, slippage)
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> size_template = vbt.RepEval('np.array([[1 / group_lens[0]]])')

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb,
            ...     size_template,  # order_args as *args
            ...     vbt.Rep('price'),
            ...     vbt.Rep('size_type'),
            ...     vbt.Rep('direction'),
            ...     vbt.Rep('fees'),
            ...     vbt.Rep('fixed_fees'),
            ...     vbt.Rep('slippage'),
            ...     segment_mask=2,  # rebalance every second tick
            ...     pre_group_func_nb=pre_group_func_nb,
            ...     pre_segment_func_nb=pre_segment_func_nb,
            ...     pre_segment_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction')
            ...     ),
            ...     broadcast_named_args=dict(  # broadcast against each other
            ...         price=close,
            ...         size_type=vbt.pf_enums.SizeType.TargetPercent,
            ...         direction=vbt.pf_enums.Direction.LongOnly,
            ...         fees=0.001,
            ...         fixed_fees=1.,
            ...         slippage=0.001
            ...     ),
            ...     template_context=dict(np=np),  # required by size_template
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/simulate_nb_example.svg)

            Templates are a very powerful tool to prepare any custom arguments after they are broadcast and
            before they are passed to the simulation function. In the example above, we use `broadcast_named_args`
            to broadcast some arguments against each other and templates to pass those objects to callbacks.
            Additionally, we used an evaluation template to compute the size based on the number of assets in each group.

            You may ask: why should we bother using broadcasting and templates if we could just pass `size=1/3`?
            Because of flexibility those features provide: we can now pass whatever parameter combinations we want
            and it will work flawlessly. For example, to create two groups of equally-allocated positions,
            we need to change only two parameters:

            ```pycon
            >>> close = np.random.uniform(1, 10, size=(5, 6))  # 6 columns instead of 3
            >>> group_by = ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']  # 2 groups instead of 1
            >>> # Replace close and group_by in the example above

            >>> pf['g1'].get_asset_value(group_by=False).vbt.plot()
            >>> pf['g2'].get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_g1.svg)

            ![](/assets/images/api/from_order_func_g2.svg)

            * Combine multiple exit conditions. Exit early if the price hits some threshold before an actual exit:

            ```pycon
            >>> @njit
            ... def pre_sim_func_nb(c):
            ...     # We need to define stop price per column once
            ...     stop_price = np.full(c.target_shape[1], np.nan, dtype=np.float_)
            ...     return (stop_price,)

            >>> @njit
            ... def order_func_nb(c, stop_price, entries, exits, size):
            ...     # Select info related to this order
            ...     entry_now = vbt.pf_nb.select_nb(c, entries)
            ...     exit_now = vbt.pf_nb.select_nb(c, exits)
            ...     size_now = vbt.pf_nb.select_nb(c, size)
            ...     price_now = vbt.pf_nb.select_nb(c, c.close)
            ...     stop_price_now = stop_price[c.col]
            ...
            ...     # Our logic
            ...     if entry_now:
            ...         if c.position_now == 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     elif exit_now or price_now >= stop_price_now:
            ...         if c.position_now > 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=-size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     return vbt.pf_enums.NoOrder

            >>> @njit
            ... def post_order_func_nb(c, stop_price, stop):
            ...     # Same broadcasting as for size
            ...     stop_now = vbt.pf_nb.select_nb(c, stop)
            ...
            ...     if c.order_result.status == vbt.pf_enums.OrderStatus.Filled:
            ...         if c.order_result.side == vbt.pf_enums.OrderSide.Buy:
            ...             # Position entered: Set stop condition
            ...             stop_price[c.col] = (1 + stop_now) * c.order_result.price
            ...         else:
            ...             # Position exited: Remove stop condition
            ...             stop_price[c.col] = np.nan

            >>> def simulate(close, entries, exits, size, stop):
            ...     return vbt.Portfolio.from_order_func(
            ...         close,
            ...         order_func_nb,
            ...         vbt.Rep('entries'), vbt.Rep('exits'), vbt.Rep('size'),  # order_args
            ...         pre_sim_func_nb=pre_sim_func_nb,
            ...         post_order_func_nb=post_order_func_nb,
            ...         post_order_args=(vbt.Rep('stop'),),
            ...         broadcast_named_args=dict(  # broadcast against each other
            ...             entries=entries,
            ...             exits=exits,
            ...             size=size,
            ...             stop=stop
            ...         )
            ...     )

            >>> close = pd.Series([10, 11, 12, 13, 14])
            >>> entries = pd.Series([True, True, False, False, False])
            >>> exits = pd.Series([False, False, False, True, True])
            >>> simulate(close, entries, exits, np.inf, 0.1).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, 0.2).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, np.nan).asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0
            4     0.0
            dtype: float64
            ```

            The reason why stop of 10% does not result in an order at the second time step is because
            it comes at the same time as entry, so it must wait until no entry is present.
            This can be changed by replacing the statement "elif" with "if", which would execute
            an exit regardless if an entry is present (similar to using `ConflictMode.Opposite` in
            `Portfolio.from_signals`).

            We can also test the parameter combinations above all at once (thanks to broadcasting
            using `vectorbtpro.base.reshaping.broadcast`):

            ```pycon
            >>> stop = pd.DataFrame([[0.1, 0.2, np.nan]])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
                  0     1     2
            0  10.0  10.0  10.0
            1   0.0   0.0   0.0
            2 -10.0 -10.0   0.0
            3   0.0   0.0 -10.0
            4   0.0   0.0   0.0
            ```

            Or much simpler using Cartesian product:

            ```pycon
            >>> stop = pd.Index([0.1, 0.2, np.nan])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
            threshold   0.1   0.2   NaN
            0          10.0  10.0  10.0
            1           0.0   0.0   0.0
            2         -10.0 -10.0   0.0
            3           0.0   0.0 -10.0
            4           0.0   0.0   0.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            * Let's illustrate how to generate multiple orders per symbol and bar.
            For each bar, buy at open and sell at close:

            ```pycon
            >>> @njit
            ... def flex_order_func_nb(c, size):
            ...     if c.call_idx == 0:
            ...         return c.from_col, vbt.pf_nb.order_nb(size=size, price=c.open[c.i, c.from_col])
            ...     if c.call_idx == 1:
            ...         return c.from_col, vbt.pf_nb.close_position_nb(price=c.close[c.i, c.from_col])
            ...     return -1, vbt.pf_enums.NoOrder

            >>> open = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> close = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 4, 5]})
            >>> size = 1
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     flex_order_func_nb, size,
            ...     open=open,
            ...     flexible=True,
            ...     max_orders=close.shape[0] * 2)

            >>> pf.orders.records_readable
                Order Id Column  Timestamp  Size  Price  Fees  Side
            0          0      a          0   1.0    1.0   0.0   Buy
            1          1      a          0   1.0    2.0   0.0  Sell
            2          2      a          1   1.0    2.0   0.0   Buy
            3          3      a          1   1.0    3.0   0.0  Sell
            4          4      a          2   1.0    3.0   0.0   Buy
            5          5      a          2   1.0    4.0   0.0  Sell
            6          0      b          0   1.0    4.0   0.0   Buy
            7          1      b          0   1.0    3.0   0.0  Sell
            8          2      b          1   1.0    5.0   0.0   Buy
            9          3      b          1   1.0    4.0   0.0  Sell
            10         4      b          2   1.0    6.0   0.0   Buy
            11         5      b          2   1.0    5.0   0.0  Sell
            ```

            !!! warning
                Each bar is effectively a black box - we don't know how the price moves in-between.
                Since trades should come in an order that closely replicates that of the real world, the only
                pieces of information that always remain in the correct order are the opening and closing price.
        """
        # Get defaults
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if isinstance(close, Data):
            data = close
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            if open is None:
                open = data.open
            if high is None:
                high = data.high
            if low is None:
                low = data.low
        if open is None:
            open_none = True
            open = np.nan
        else:
            open_none = False
        if high is None:
            high_none = True
            high = np.nan
        else:
            high_none = False
        if low is None:
            low_none = True
            low = np.nan
        else:
            low_none = False

        if flexible is None:
            flexible = portfolio_cfg["flexible"]
        if init_cash is None:
            init_cash = portfolio_cfg["init_cash"]
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if checks.is_int(init_cash) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg["init_position"]
        if init_price is None:
            init_price = portfolio_cfg["init_price"]
        if cash_deposits is None:
            cash_deposits = portfolio_cfg["cash_deposits"]
        if cash_earnings is None:
            cash_earnings = portfolio_cfg["cash_earnings"]
        if cash_sharing is None:
            cash_sharing = portfolio_cfg["cash_sharing"]
        if cash_sharing and group_by is None:
            group_by = True
        if not flexible:
            if call_seq is None:
                call_seq = portfolio_cfg["call_seq"]
            call_seq = map_enum_fields(call_seq, CallSeqType)
            if checks.is_int(call_seq):
                if call_seq == CallSeqType.Auto:
                    raise ValueError(
                        "CallSeqType.Auto must be implemented manually. Use sort_call_seq_1d_nb in pre_segment_func_nb."
                    )
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg["attach_call_seq"]
        if segment_mask is None:
            segment_mask = True
        if call_pre_segment is None:
            call_pre_segment = portfolio_cfg["call_pre_segment"]
        if call_post_segment is None:
            call_post_segment = portfolio_cfg["call_post_segment"]
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg["ffill_val_price"]
        if update_value is None:
            update_value = portfolio_cfg["update_value"]
        if fill_pos_record is None:
            fill_pos_record = portfolio_cfg["fill_pos_record"]
        if track_value is None:
            track_value = portfolio_cfg["track_value"]
        if row_wise is None:
            row_wise = portfolio_cfg["row_wise"]
        if seed is None:
            seed = portfolio_cfg["seed"]
        if seed is not None:
            set_seed(seed)
        if in_outputs is not None and not checks.is_namedtuple(in_outputs):
            in_outputs = to_mapping(in_outputs)
            in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        if group_by is None:
            group_by = portfolio_cfg["group_by"]
        if freq is None:
            freq = portfolio_cfg["freq"]
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_kwargs = merge_dicts(portfolio_cfg["broadcast_kwargs"], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get("require_kwargs", {})
        template_context = merge_dicts(portfolio_cfg["template_context"], template_context)
        if keep_inout_raw is None:
            keep_inout_raw = portfolio_cfg["keep_inout_raw"]
        if template_context is None:
            template_context = {}
        if bm_close is None:
            bm_close = portfolio_cfg["bm_close"]

        # Prepare the simulation
        broadcastable_args = dict(cash_earnings=cash_earnings, open=open, high=high, low=low, close=close)
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args["bm_close"] = bm_close
        else:
            broadcastable_args["bm_close"] = np.nan
        broadcastable_args = {**broadcastable_args, **broadcast_named_args}
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=True,
                reindex_kwargs=dict(
                    cash_earnings=dict(fill_value=0.0),
                    open=dict(fill_value=np.nan),
                    high=dict(fill_value=np.nan),
                    low=dict(fill_value=np.nan),
                    close=dict(fill_value=np.nan),
                    bm_close=dict(fill_value=np.nan),
                ),
                wrapper_kwargs=dict(
                    freq=freq,
                    group_by=group_by,
                ),
            ),
            broadcast_kwargs,
        )
        broadcasted_args, wrapper = broadcast(broadcastable_args, return_wrapper=True, **broadcast_kwargs)
        if not wrapper.group_select and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        cash_earnings = broadcasted_args.pop("cash_earnings")
        target_shape_2d = wrapper.shape_2d
        index = wrapper.ns_index
        freq = wrapper.ns_freq

        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        init_price = np.require(np.broadcast_to(init_price, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            cash_deposits,
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            keep_flex=keep_inout_raw,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=require_kwargs,
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_int(segment_mask):
            if keep_inout_raw:
                _segment_mask = np.full((target_shape_2d[0], 1), False)
            else:
                _segment_mask = np.full((target_shape_2d[0], len(group_lens)), False)
            _segment_mask[0::segment_mask] = True
            segment_mask = _segment_mask
        else:
            segment_mask = broadcast(
                segment_mask,
                to_shape=(target_shape_2d[0], len(group_lens)),
                keep_flex=keep_inout_raw,
                reindex_kwargs=dict(fill_value=False),
                require_kwargs=require_kwargs,
            )
        if not flexible:
            if call_seq is None and attach_call_seq:
                call_seq = CallSeqType.Default
            if call_seq is not None:
                if checks.is_any_array(call_seq):
                    call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
                else:
                    call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)

        # Check data types
        checks.assert_subdtype(cs_group_lens, np.integer)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(init_price, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(segment_mask, np.bool_)
        checks.assert_subdtype(broadcasted_args["open"], np.number)
        checks.assert_subdtype(broadcasted_args["high"], np.number)
        checks.assert_subdtype(broadcasted_args["low"], np.number)
        checks.assert_subdtype(broadcasted_args["close"], np.number)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args["bm_close"], np.number)

        # Prepare arguments
        template_context = merge_dicts(
            broadcasted_args,
            dict(
                target_shape=target_shape_2d,
                index=index,
                freq=freq,
                group_lens=group_lens,
                cs_group_lens=cs_group_lens,
                cash_sharing=cash_sharing,
                init_cash=init_cash,
                init_position=init_position,
                init_price=init_price,
                cash_deposits=cash_deposits,
                cash_earnings=cash_earnings,
                segment_mask=segment_mask,
                call_pre_segment=call_pre_segment,
                call_post_segment=call_post_segment,
                pre_sim_func_nb=pre_sim_func_nb,
                pre_sim_args=pre_sim_args,
                post_sim_func_nb=post_sim_func_nb,
                post_sim_args=post_sim_args,
                pre_group_func_nb=pre_group_func_nb,
                pre_group_args=pre_group_args,
                post_group_func_nb=post_group_func_nb,
                post_group_args=post_group_args,
                pre_row_func_nb=pre_row_func_nb,
                pre_row_args=pre_row_args,
                post_row_func_nb=post_row_func_nb,
                post_row_args=post_row_args,
                pre_segment_func_nb=pre_segment_func_nb,
                pre_segment_args=pre_segment_args,
                post_segment_func_nb=post_segment_func_nb,
                post_segment_args=post_segment_args,
                flex_order_func_nb=order_func_nb,
                flex_order_args=order_args,
                post_order_func_nb=post_order_func_nb,
                post_order_args=post_order_args,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                fill_pos_record=fill_pos_record,
                track_value=track_value,
                max_orders=max_orders,
                max_logs=max_logs,
                in_outputs=in_outputs,
                wrapper=wrapper,
            ),
            template_context,
        )
        pre_sim_args = deep_substitute(pre_sim_args, template_context, sub_id="pre_sim_args")
        post_sim_args = deep_substitute(post_sim_args, template_context, sub_id="post_sim_args")
        pre_group_args = deep_substitute(pre_group_args, template_context, sub_id="pre_group_args")
        post_group_args = deep_substitute(post_group_args, template_context, sub_id="post_group_args")
        pre_row_args = deep_substitute(pre_row_args, template_context, sub_id="pre_row_args")
        post_row_args = deep_substitute(post_row_args, template_context, sub_id="post_row_args")
        pre_segment_args = deep_substitute(pre_segment_args, template_context, sub_id="pre_segment_args")
        post_segment_args = deep_substitute(post_segment_args, template_context, sub_id="post_segment_args")
        order_args = deep_substitute(order_args, template_context, sub_id="order_args")
        post_order_args = deep_substitute(post_order_args, template_context, sub_id="post_order_args")
        in_outputs = deep_substitute(in_outputs, template_context, sub_id="in_outputs")
        for k in broadcast_named_args:
            broadcasted_args.pop(k)

        # Perform the simulation
        if row_wise:
            if flexible:
                func = jit_reg.resolve_option(nb.flex_simulate_row_wise_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                sim_out = func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_row_func_nb=pre_row_func_nb,
                    pre_row_args=pre_row_args,
                    post_row_func_nb=post_row_func_nb,
                    post_row_args=post_row_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    flex_order_func_nb=order_func_nb,
                    flex_order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    index=index,
                    freq=freq,
                    open=broadcasted_args["open"],
                    high=broadcasted_args["high"],
                    low=broadcasted_args["low"],
                    close=broadcasted_args["close"],
                    bm_close=broadcasted_args["bm_close"],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    in_outputs=in_outputs,
                )
            else:
                func = jit_reg.resolve_option(nb.simulate_row_wise_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                sim_out = func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_row_func_nb=pre_row_func_nb,
                    pre_row_args=pre_row_args,
                    post_row_func_nb=post_row_func_nb,
                    post_row_args=post_row_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    order_func_nb=order_func_nb,
                    order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    index=index,
                    freq=freq,
                    open=broadcasted_args["open"],
                    high=broadcasted_args["high"],
                    low=broadcasted_args["low"],
                    close=broadcasted_args["close"],
                    bm_close=broadcasted_args["bm_close"],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    in_outputs=in_outputs,
                )
        else:
            if flexible:
                func = jit_reg.resolve_option(nb.flex_simulate_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                sim_out = func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_group_func_nb=pre_group_func_nb,
                    pre_group_args=pre_group_args,
                    post_group_func_nb=post_group_func_nb,
                    post_group_args=post_group_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    flex_order_func_nb=order_func_nb,
                    flex_order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    index=index,
                    freq=freq,
                    open=broadcasted_args["open"],
                    high=broadcasted_args["high"],
                    low=broadcasted_args["low"],
                    close=broadcasted_args["close"],
                    bm_close=broadcasted_args["bm_close"],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    in_outputs=in_outputs,
                )
            else:
                func = jit_reg.resolve_option(nb.simulate_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                sim_out = func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_group_func_nb=pre_group_func_nb,
                    pre_group_args=pre_group_args,
                    post_group_func_nb=post_group_func_nb,
                    post_group_args=post_group_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    order_func_nb=order_func_nb,
                    order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    index=index,
                    freq=freq,
                    open=broadcasted_args["open"],
                    high=broadcasted_args["high"],
                    low=broadcasted_args["low"],
                    close=broadcasted_args["close"],
                    bm_close=broadcasted_args["bm_close"],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    in_outputs=in_outputs,
                )

        # Create an instance
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = broadcasted_args["bm_close"]
        return cls(
            wrapper,
            broadcasted_args["close"],
            sim_out.order_records,
            open=broadcasted_args["open"] if not open_none else None,
            high=broadcasted_args["high"] if not high_none else None,
            low=broadcasted_args["low"] if not low_none else None,
            log_records=sim_out.log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=sim_out.cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if not flexible and attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs,
        )

    @classmethod
    def from_def_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        lock_cash: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        pre_segment_func_nb: tp.Optional[nb.PreSegmentFuncT] = None,
        order_func_nb: tp.Optional[tp.Union[nb.OrderFuncT, nb.FlexOrderFuncT]] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        flexible: tp.Optional[bool] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PortfolioT:
        """Build portfolio from the default order function.

        Default order function takes size, price, fees, and other available information, and issues
        an order at each column and time step. Additionally, it uses a segment preprocessing function
        that overrides the valuation price and sorts the call sequence. This way, it behaves similarly to
        `Portfolio.from_orders`, but allows injecting pre- and postprocessing functions to have more
        control over the execution. It also knows how to chunk each argument. The only disadvantage is
        that `Portfolio.from_orders` is more optimized towards performance (up to 5x).

        If `flexible` is True:

        * `pre_segment_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_flex_pre_segment_func_nb`.
        * `order_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_flex_order_func_nb`.

        If `flexible` is False:

        * Pre-segment function is `vectorbtpro.portfolio.nb.from_order_func.def_pre_segment_func_nb`.
        * Order function is `vectorbtpro.portfolio.nb.from_order_func.def_order_func_nb`.

        For details on other arguments, see `Portfolio.from_orders` and `Portfolio.from_order_func`.

        Usage:
            * Working with `Portfolio.from_def_order_func` is a similar experience as working
            with `Portfolio.from_orders`:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_def_order_func(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Equal-weighted portfolio as in the example under `Portfolio.from_order_func`
            but much less verbose and with asset value pre-computed during the simulation (= faster):

            ```pycon
            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))

            >>> @njit
            ... def post_segment_func_nb(c):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.in_outputs.asset_value_pc[c.i, col] = c.last_position[col] * c.last_val_price[col]

            >>> pf = vbt.Portfolio.from_def_order_func(
            ...     close,
            ...     size=1/3,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     fees=0.001,
            ...     fixed_fees=1.,
            ...     slippage=0.001,
            ...     segment_mask=2,
            ...     cash_sharing=True,
            ...     group_by=True,
            ...     call_seq='auto',
            ...     post_segment_func_nb=post_segment_func_nb,
            ...     call_post_segment=True,
            ...     in_outputs=dict(asset_value_pc=vbt.RepEval('np.empty_like(close)'))
            ... )

            >>> asset_value = pf.wrapper.wrap(pf.in_outputs.asset_value_pc, group_by=False)
            >>> asset_value.vbt.plot().show()
            ```

            ![](/assets/images/api/simulate_nb_example.svg)
        """
        # Get defaults
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if flexible is None:
            flexible = portfolio_cfg["flexible"]
        if size is None:
            size = portfolio_cfg["size"]
        if size_type is None:
            size_type = portfolio_cfg["size_type"]
        if direction is None:
            direction = portfolio_cfg["direction"]
        if price is None:
            price = portfolio_cfg["price"]
        if size is None:
            size = portfolio_cfg["size"]
        if fees is None:
            fees = portfolio_cfg["fees"]
        if fixed_fees is None:
            fixed_fees = portfolio_cfg["fixed_fees"]
        if slippage is None:
            slippage = portfolio_cfg["slippage"]
        if min_size is None:
            min_size = portfolio_cfg["min_size"]
        if max_size is None:
            max_size = portfolio_cfg["max_size"]
        if size_granularity is None:
            size_granularity = portfolio_cfg["size_granularity"]
        if reject_prob is None:
            reject_prob = portfolio_cfg["reject_prob"]
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg["price_area_vio_mode"]
        if lock_cash is None:
            lock_cash = portfolio_cfg["lock_cash"]
        if allow_partial is None:
            allow_partial = portfolio_cfg["allow_partial"]
        if raise_reject is None:
            raise_reject = portfolio_cfg["raise_reject"]
        if log is None:
            log = portfolio_cfg["log"]
        if val_price is None:
            val_price = portfolio_cfg["val_price"]
        if call_seq is None:
            call_seq = portfolio_cfg["call_seq"]
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if checks.is_int(call_seq):
            if call_seq == CallSeqType.Auto:
                auto_call_seq = True
                call_seq = None
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_named_args = merge_dicts(
            dict(
                size=size,
                size_type=size_type,
                direction=direction,
                price=price,
                fees=fees,
                fixed_fees=fixed_fees,
                slippage=slippage,
                min_size=min_size,
                max_size=max_size,
                size_granularity=size_granularity,
                reject_prob=reject_prob,
                price_area_vio_mode=price_area_vio_mode,
                lock_cash=lock_cash,
                allow_partial=allow_partial,
                raise_reject=raise_reject,
                log=log,
                val_price=val_price,
            ),
            broadcast_named_args,
        )
        broadcast_kwargs = merge_dicts(
            portfolio_cfg["broadcast_kwargs"],
            dict(
                reindex_kwargs=dict(
                    size=dict(fill_value=np.nan),
                    price=dict(fill_value=np.nan),
                    size_type=dict(fill_value=SizeType.Amount),
                    direction=dict(fill_value=Direction.Both),
                    fees=dict(fill_value=0.0),
                    fixed_fees=dict(fill_value=0.0),
                    slippage=dict(fill_value=0.0),
                    min_size=dict(fill_value=np.nan),
                    max_size=dict(fill_value=np.nan),
                    size_granularity=dict(fill_value=np.nan),
                    reject_prob=dict(fill_value=0.0),
                    price_area_vio_mode=dict(fill_value=PriceAreaVioMode.Ignore),
                    lock_cash=dict(fill_value=False),
                    allow_partial=dict(fill_value=True),
                    raise_reject=dict(fill_value=False),
                    log=dict(fill_value=False),
                    val_price=dict(val_price=np.nan),
                )
            ),
            broadcast_kwargs,
        )

        # Prepare arguments and pass to from_order_func

        def _prepare_size(size):
            checks.assert_subdtype(size, np.number)
            return size

        def _prepare_price(price):
            price = map_enum_fields(price, PriceType, ignore_type=(int, float))
            checks.assert_subdtype(price, np.number)
            return price

        def _prepare_size_type(size_type):
            size_type = map_enum_fields(size_type, SizeType)
            checks.assert_subdtype(size_type, np.integer)
            return size_type

        def _prepare_direction(direction):
            direction = map_enum_fields(direction, Direction)
            checks.assert_subdtype(direction, np.integer)
            return direction

        def _prepare_fees(fees):
            checks.assert_subdtype(fees, np.number)
            return fees

        def _prepare_fixed_fees(fixed_fees):
            checks.assert_subdtype(fixed_fees, np.number)
            return fixed_fees

        def _prepare_slippage(slippage):
            checks.assert_subdtype(slippage, np.number)
            return slippage

        def _prepare_min_size(min_size):
            checks.assert_subdtype(min_size, np.number)
            return min_size

        def _prepare_max_size(max_size):
            checks.assert_subdtype(max_size, np.number)
            return max_size

        def _prepare_size_granularity(size_granularity):
            checks.assert_subdtype(size_granularity, np.number)
            return size_granularity

        def _prepare_reject_prob(reject_prob):
            checks.assert_subdtype(reject_prob, np.number)
            return reject_prob

        def _prepare_price_area_vio_mode(price_area_vio_mode):
            price_area_vio_mode = map_enum_fields(price_area_vio_mode, PriceAreaVioMode)
            checks.assert_subdtype(price_area_vio_mode, np.integer)
            return price_area_vio_mode

        def _prepare_lock_cash(lock_cash):
            checks.assert_subdtype(lock_cash, np.bool_)
            return lock_cash

        def _prepare_allow_partial(allow_partial):
            checks.assert_subdtype(allow_partial, np.bool_)
            return allow_partial

        def _prepare_raise_reject(raise_reject):
            checks.assert_subdtype(raise_reject, np.bool_)
            return raise_reject

        def _prepare_log(log):
            checks.assert_subdtype(log, np.bool_)
            return log

        def _prepare_val_price(val_price):
            val_price = map_enum_fields(val_price, ValPriceType, ignore_type=(int, float))
            checks.assert_subdtype(val_price, np.number)
            return val_price

        if flexible:
            if pre_segment_func_nb is None:
                pre_segment_func_nb = nb.def_flex_pre_segment_func_nb
            if order_func_nb is None:
                order_func_nb = nb.def_flex_order_func_nb
        else:
            if pre_segment_func_nb is None:
                pre_segment_func_nb = nb.def_pre_segment_func_nb
            if order_func_nb is None:
                order_func_nb = nb.def_order_func_nb
        order_args = (
            RepFunc(_prepare_size),
            RepFunc(_prepare_price),
            RepFunc(_prepare_size_type),
            RepFunc(_prepare_direction),
            RepFunc(_prepare_fees),
            RepFunc(_prepare_fixed_fees),
            RepFunc(_prepare_slippage),
            RepFunc(_prepare_min_size),
            RepFunc(_prepare_max_size),
            RepFunc(_prepare_size_granularity),
            RepFunc(_prepare_reject_prob),
            RepFunc(_prepare_price_area_vio_mode),
            RepFunc(_prepare_lock_cash),
            RepFunc(_prepare_allow_partial),
            RepFunc(_prepare_raise_reject),
            RepFunc(_prepare_log),
        )
        pre_segment_args = (
            RepFunc(_prepare_val_price),
            RepFunc(_prepare_price),
            RepFunc(_prepare_size),
            RepFunc(_prepare_size_type),
            RepFunc(_prepare_direction),
            auto_call_seq,
        )
        arg_take_spec = dict(
            pre_segment_args=ch.ArgsTaker(
                *[base_ch.flex_array_gl_slicer if isinstance(x, RepFunc) else None for x in pre_segment_args],
            )
        )
        order_args_taker = ch.ArgsTaker(
            *[base_ch.flex_array_gl_slicer if isinstance(x, RepFunc) else None for x in order_args],
        )
        if flexible:
            arg_take_spec["flex_order_args"] = order_args_taker
        else:
            arg_take_spec["order_args"] = order_args_taker
        chunked = ch.specialize_chunked_option(chunked, arg_take_spec=arg_take_spec)
        return cls.from_order_func(
            close,
            order_func_nb,
            *order_args,
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=pre_segment_args,
            flexible=flexible,
            call_seq=call_seq,
            broadcast_named_args=broadcast_named_args,
            chunked=chunked,
            **kwargs,
        )

    # ############# Grouping ############# #

    def regroup(self: PortfolioT, group_by: tp.GroupByLike, **kwargs) -> PortfolioT:
        """Regroup this object.

        See `vectorbtpro.base.wrapping.Wrapping.regroup`.

        !!! note
            All cached objects will be lost."""
        if self.cash_sharing:
            if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                raise ValueError("Cannot modify grouping globally when cash_sharing=True")
        return Wrapping.regroup(self, group_by, **kwargs)

    # ############# Properties ############# #

    @property
    def cash_sharing(self) -> bool:
        """Whether to share cash within the same group."""
        return self._cash_sharing

    @property
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Named tuple with in-output objects."""
        return self._in_outputs

    @property
    def use_in_outputs(self) -> bool:
        """Whether to return in-output objects when calling properties."""
        return self._use_in_outputs

    @property
    def fillna_close(self) -> bool:
        """Whether to forward-backward fill NaN values in `Portfolio.close`."""
        return self._fillna_close

    @property
    def trades_type(self) -> int:
        """Default `vectorbtpro.portfolio.trades.Trades` to use across `Portfolio`."""
        return self._trades_type

    @property
    def orders_cls(self) -> type:
        """Class for wrapping order records."""
        return self._orders_cls

    @property
    def logs_cls(self) -> type:
        """Class for wrapping log records."""
        return self._logs_cls

    @property
    def trades_cls(self) -> type:
        """Class for wrapping trade records."""
        return self._trades_cls

    @property
    def entry_trades_cls(self) -> type:
        """Class for wrapping entry trade records."""
        return self._entry_trades_cls

    @property
    def exit_trades_cls(self) -> type:
        """Class for wrapping exit trade records."""
        return self._exit_trades_cls

    @property
    def positions_cls(self) -> type:
        """Class for wrapping position records."""
        return self._positions_cls

    @property
    def drawdowns_cls(self) -> type:
        """Class for wrapping drawdown records."""
        return self._drawdowns_cls

    @custom_property(group_by_aware=False)
    def call_seq(self) -> tp.Optional[tp.SeriesFrame]:
        """Sequence of calls per row and group."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "call_seq"):
            call_seq = self.in_outputs.call_seq
        else:
            call_seq = self._call_seq
        if call_seq is None:
            return None

        return self.wrapper.wrap(call_seq, group_by=False)

    # ############# Price ############# #

    @custom_property(group_by_aware=False, resample_func="first")
    def open(self) -> tp.SeriesFrame:
        """Open price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "open"):
            open = self.in_outputs.open
        else:
            open = self._open

        if open is None:
            return None
        return self.wrapper.wrap(open, group_by=False)

    @custom_property(group_by_aware=False, resample_func="max")
    def high(self) -> tp.SeriesFrame:
        """High price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "high"):
            high = self.in_outputs.high
        else:
            high = self._high

        if high is None:
            return None
        return self.wrapper.wrap(high, group_by=False)

    @custom_property(group_by_aware=False, resample_func="min")
    def low(self) -> tp.SeriesFrame:
        """Low price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "low"):
            low = self.in_outputs.low
        else:
            low = self._low

        if low is None:
            return None
        return self.wrapper.wrap(low, group_by=False)

    @custom_property(group_by_aware=False, resample_func="last")
    def close(self) -> tp.SeriesFrame:
        """Last asset price at each time step."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "close"):
            close = self.in_outputs.close
        else:
            close = self._close

        return self.wrapper.wrap(close, group_by=False)

    @class_or_instancemethod
    def get_filled_close(
        cls_or_self,
        close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get forward and backward filled closing price.

        See `vectorbtpro.generic.nb.base.fbfill_nb`."""
        if not isinstance(cls_or_self, type):
            if close is None:
                close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_close = func(to_2d_array(close))
        return wrapper.wrap(filled_close, group_by=False, **resolve_dict(wrap_kwargs))

    @custom_property(group_by_aware=False, resample_func="last")
    def bm_close(self) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Benchmark price per unit series."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "bm_close"):
            bm_close = self.in_outputs.bm_close
        else:
            bm_close = self._bm_close

        if bm_close is None or isinstance(bm_close, bool):
            return bm_close
        return self.wrapper.wrap(bm_close, group_by=False)

    @class_or_instancemethod
    def get_filled_bm_close(
        cls_or_self,
        bm_close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Get forward and backward filled benchmark closing price.

        See `vectorbtpro.generic.nb.base.fbfill_nb`."""
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if bm_close is None or isinstance(bm_close, bool):
                    return bm_close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(bm_close)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_bm_close = func(to_2d_array(bm_close))
        return wrapper.wrap(filled_bm_close, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Records ############# #

    @property
    def order_records(self) -> tp.RecordArray:
        """A structured NumPy array of order records."""
        return self._order_records

    @class_or_instancemethod
    def get_orders(
        cls_or_self,
        order_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> Orders:
        """Get order records.

        See `vectorbtpro.portfolio.orders.Orders`."""
        if not isinstance(cls_or_self, type):
            if order_records is None:
                order_records = cls_or_self.order_records
            if open is None:
                open = cls_or_self._open
            if high is None:
                high = cls_or_self._high
            if low is None:
                low = cls_or_self._low
            if close is None:
                close = cls_or_self._close
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
            if orders_cls is None:
                orders_cls = cls_or_self.orders_cls
        else:
            checks.assert_not_none(order_records)
            checks.assert_not_none(wrapper)
            if orders_cls is None:
                orders_cls = Orders

        return orders_cls(
            wrapper,
            order_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @property
    def log_records(self) -> tp.RecordArray:
        """A structured NumPy array of log records."""
        return self._log_records

    @class_or_instancemethod
    def get_logs(
        cls_or_self,
        log_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        logs_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> Logs:
        """Get log records.

        See `vectorbtpro.portfolio.logs.Logs`."""
        if not isinstance(cls_or_self, type):
            if log_records is None:
                log_records = cls_or_self.log_records
            if open is None:
                open = cls_or_self._open
            if high is None:
                high = cls_or_self._high
            if low is None:
                low = cls_or_self._low
            if close is None:
                close = cls_or_self._close
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
            if logs_cls is None:
                logs_cls = cls_or_self.logs_cls
        else:
            checks.assert_not_none(log_records)
            checks.assert_not_none(wrapper)
            if logs_cls is None:
                logs_cls = Logs

        return logs_cls(
            wrapper,
            log_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_entry_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        entry_trades_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> EntryTrades:
        """Get entry trade records.

        See `vectorbtpro.portfolio.trades.EntryTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if entry_trades_cls is None:
                entry_trades_cls = cls_or_self.entry_trades_cls
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if entry_trades_cls is None:
                entry_trades_cls = EntryTrades

        return entry_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_exit_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        exit_trades_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> ExitTrades:
        """Get exit trade records.

        See `vectorbtpro.portfolio.trades.ExitTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if exit_trades_cls is None:
                exit_trades_cls = cls_or_self.exit_trades_cls
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if exit_trades_cls is None:
                exit_trades_cls = ExitTrades

        return exit_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_positions(
        cls_or_self,
        trades: tp.Optional[Trades] = None,
        positions_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Positions:
        """Get position records.

        See `vectorbtpro.portfolio.trades.Positions`."""
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.exit_trades
            if positions_cls is None:
                positions_cls = cls_or_self.positions_cls
        else:
            checks.assert_not_none(trades)
            if positions_cls is None:
                positions_cls = Positions

        return positions_cls.from_trades(trades, **kwargs).regroup(group_by)

    def get_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> Trades:
        """Get trade/position records depending upon `Portfolio.trades_type`."""
        if self.trades_type == TradesType.Trades:
            raise NotImplementedError
        if self.trades_type == TradesType.EntryTrades:
            return self.resolve_shortcut_attr("entry_trades", group_by=group_by, **kwargs)
        if self.trades_type == TradesType.ExitTrades:
            return self.resolve_shortcut_attr("exit_trades", group_by=group_by, **kwargs)
        return self.resolve_shortcut_attr("positions", group_by=group_by, **kwargs)

    def get_trade_history(
        self,
        orders: tp.Optional[Orders] = None,
        entry_trades_cls: tp.Optional[type] = None,
        exit_trades_cls: tp.Optional[type] = None,
        **kwargs,
    ) -> tp.Frame:
        """Get (entry and exit) trade history as a DataFrame."""
        if orders is None:
            orders = self.orders
        entry_trades = self.resolve_shortcut_attr(
            "entry_trades", orders=orders, entry_trades_cls=entry_trades_cls, **kwargs
        )
        exit_trades = self.resolve_shortcut_attr(
            "exit_trades", orders=orders, exit_trades_cls=exit_trades_cls, **kwargs
        )

        order_history = orders.records_readable
        del order_history["Size"]
        del order_history["Price"]
        del order_history["Fees"]
        entry_trade_history = entry_trades.records_readable
        exit_trade_history = exit_trades.records_readable
        entry_trade_history.rename(columns={"Entry Order Id": "Order Id"}, inplace=True)
        exit_trade_history.rename(columns={"Exit Order Id": "Order Id"}, inplace=True)
        del entry_trade_history["Entry Index"]
        del exit_trade_history["Exit Index"]
        entry_trade_history.rename(columns={"Avg Entry Price": "Price"}, inplace=True)
        exit_trade_history.rename(columns={"Avg Exit Price": "Price"}, inplace=True)
        entry_trade_history.rename(columns={"Entry Fees": "Fees"}, inplace=True)
        exit_trade_history.rename(columns={"Exit Fees": "Fees"}, inplace=True)
        del entry_trade_history["Exit Order Id"]
        del exit_trade_history["Entry Order Id"]
        del entry_trade_history["Exit Index"]
        del exit_trade_history["Entry Index"]
        del entry_trade_history["Avg Exit Price"]
        del exit_trade_history["Avg Entry Price"]
        del entry_trade_history["Exit Fees"]
        del exit_trade_history["Entry Fees"]

        trade_history = pd.concat((entry_trade_history, exit_trade_history), axis=0)
        trade_history = pd.merge(order_history, trade_history, on=["Column", "Order Id"])
        trade_history = trade_history.sort_values(by=["Column", "Order Id", "Position Id"])
        trade_history["Entry Trade Id"] = trade_history["Entry Trade Id"].fillna(-1).astype(int)
        trade_history["Exit Trade Id"] = trade_history["Exit Trade Id"].fillna(-1).astype(int)
        trade_history["Entry Trade Id"] = trade_history.pop("Entry Trade Id")
        trade_history["Exit Trade Id"] = trade_history.pop("Exit Trade Id")
        trade_history["Position Id"] = trade_history.pop("Position Id")
        return trade_history

    @class_or_instancemethod
    def get_drawdowns(
        cls_or_self,
        value: tp.Optional[tp.SeriesFrame] = None,
        drawdowns_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Drawdowns:
        """Get drawdown records from `Portfolio.get_value`.

        See `vectorbtpro.generic.drawdowns.Drawdowns`."""
        if not isinstance(cls_or_self, type):
            if value is None:
                value = cls_or_self.resolve_shortcut_attr("value", group_by=group_by)
            wrapper_kwargs = merge_dicts(cls_or_self.orders.wrapper.config, wrapper_kwargs, dict(group_by=None))
            if drawdowns_cls is None:
                drawdowns_cls = cls_or_self.drawdowns_cls
        else:
            checks.assert_not_none(value)
            if drawdowns_cls is None:
                drawdowns_cls = Drawdowns

        return drawdowns_cls.from_price(value, wrapper_kwargs=wrapper_kwargs, **kwargs)

    # ############# Assets ############# #

    @class_or_instancemethod
    def get_init_position(
        cls_or_self,
        init_position_raw: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial position per column."""
        if not isinstance(cls_or_self, type):
            if init_position_raw is None:
                init_position_raw = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_raw)
            checks.assert_not_none(wrapper)

        init_position = np.broadcast_to(to_1d_array(init_position_raw), (wrapper.shape_2d[1],))
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_flow(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset flow series per column.

        Returns the total transacted amount of assets at each time step."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        direction = map_enum_fields(direction, Direction)
        func = jit_reg.resolve_option(nb.asset_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            direction=direction,
            init_position=to_1d_array(init_position),
        )
        return wrapper.wrap(asset_flow, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_assets(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        asset_flow: tp.Optional[tp.SeriesFrame] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset series per column.

        Returns the position at each time step."""
        if not isinstance(cls_or_self, type):
            if asset_flow is None:
                asset_flow = cls_or_self.resolve_shortcut_attr(
                    "asset_flow",
                    direction=Direction.Both,
                    jitted=jitted,
                    chunked=chunked,
                )
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_flow)
            if init_position is None:
                init_position = 0.0
            checks.assert_not_none(wrapper)

        direction = map_enum_fields(direction, Direction)
        func = jit_reg.resolve_option(nb.assets_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        assets = func(to_2d_array(asset_flow), init_position=to_1d_array(init_position))
        if direction == Direction.LongOnly:
            func = jit_reg.resolve_option(nb.longonly_assets_nb, jitted)
            assets = func(assets)
        elif direction == Direction.ShortOnly:
            func = jit_reg.resolve_option(nb.shortonly_assets_nb, jitted)
            assets = func(assets)
        return wrapper.wrap(assets, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_mask(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        assets: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get position mask per column/group.

        An element is True if there is a position at the given time step."""
        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        position_mask = to_2d_array(assets) != 0
        if wrapper.grouper.is_grouped(group_by=group_by):
            position_mask = (
                wrapper.wrap(position_mask, group_by=False)
                .vbt(wrapper=wrapper)
                .squeeze_grouped(
                    jit_reg.resolve_option(generic_nb.any_reduce_nb, jitted),
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            )
        return wrapper.wrap(position_mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_coverage(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        position_mask: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get position coverage per column/group.

        Position coverage is the number of time steps in the market divided by the total number of time steps."""
        if not isinstance(cls_or_self, type):
            if position_mask is None:
                position_mask = cls_or_self.resolve_shortcut_attr(
                    "position_mask",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                    group_by=False,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(position_mask)
            checks.assert_not_none(wrapper)

        position_coverage = position_mask.vbt(wrapper=wrapper).reduce(
            jit_reg.resolve_option(generic_nb.mean_reduce_nb, jitted),
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="position_coverage"), wrap_kwargs)
        return wrapper.wrap_reduced(position_coverage, group_by=group_by, **wrap_kwargs)

    # ############# Cash ############# #

    @class_or_instancemethod
    def get_cash_deposits(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        split_shared: bool = False,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get cash deposit series per column/group.

        Set `keep_flex` to True to keep format suitable for flexible indexing.
        This consumes less memory."""
        if not isinstance(cls_or_self, type):
            if cash_deposits_raw is None:
                cash_deposits_raw = cls_or_self._cash_deposits
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_deposits_raw is None:
                cash_deposits_raw = 0.0
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(wrapper)

        cash_deposits_raw = to_2d_array(cash_deposits_raw)
        if wrapper.grouper.is_grouped(group_by=group_by):
            if keep_flex and cash_sharing:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_deposits_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(wrapper.shape_2d, cash_deposits_raw, group_lens, cash_sharing)
        else:
            if keep_flex and not cash_sharing:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens()
            func = jit_reg.resolve_option(nb.cash_deposits_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(
                wrapper.shape_2d,
                cash_deposits_raw,
                group_lens,
                cash_sharing,
                split_shared=split_shared,
            )
        if keep_flex:
            return cash_deposits
        return wrapper.wrap(cash_deposits, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash_earnings(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_earnings_raw: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get earnings in cash series per column/group.

        Set `keep_flex` to True to keep format suitable for flexible indexing.
        This consumes less memory."""
        if not isinstance(cls_or_self, type):
            if cash_earnings_raw is None:
                cash_earnings_raw = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_earnings_raw is None:
                cash_earnings_raw = 0.0
            checks.assert_not_none(wrapper)

        cash_earnings_raw = to_2d_array(cash_earnings_raw)
        if wrapper.grouper.is_grouped(group_by=group_by):
            cash_earnings = np.broadcast_to(cash_earnings_raw, wrapper.shape_2d)
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_earnings = func(cash_earnings, group_lens)
        else:
            if keep_flex:
                return cash_earnings_raw
            cash_earnings = np.broadcast_to(cash_earnings_raw, wrapper.shape_2d)
        if keep_flex:
            return cash_earnings
        return wrapper.wrap(cash_earnings, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash_flow(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        free: bool = False,
        orders: tp.Optional[Orders] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash flow series per column/group.

        Use `free` to return the flow of the free cash, which never goes above the initial level,
        because an operation always costs money.

        !!! note
            Does not include cash deposits, but includes earnings."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if cash_earnings is None:
                cash_earnings = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        func = jit_reg.resolve_option(nb.cash_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        cash_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            free=free,
            cash_earnings=to_2d_array(cash_earnings),
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_flow = func(cash_flow, group_lens)
        return wrapper.wrap(cash_flow, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_init_cash(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_cash_raw: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        free_cash_flow: tp.Optional[tp.SeriesFrame] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial amount of cash per column/group."""
        if not isinstance(cls_or_self, type):
            if init_cash_raw is None:
                init_cash_raw = cls_or_self._init_cash
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash_raw)
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(wrapper)

        if checks.is_int(init_cash_raw) and init_cash_raw in InitCashMode:
            if not isinstance(cls_or_self, type):
                if free_cash_flow is None:
                    free_cash_flow = cls_or_self.resolve_shortcut_attr(
                        "cash_flow",
                        group_by=group_by,
                        free=True,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            else:
                checks.assert_not_none(free_cash_flow)
                if cash_deposits is None:
                    cash_deposits = 0.0
            func = jit_reg.resolve_option(nb.align_init_cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            init_cash = func(
                init_cash_raw,
                to_2d_array(free_cash_flow),
                cash_deposits=to_2d_array(cash_deposits),
            )
        else:
            init_cash_raw = to_1d_array(init_cash_raw)
            if wrapper.grouper.is_grouped(group_by=group_by):
                group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
                func = jit_reg.resolve_option(nb.init_cash_grouped_nb, jitted)
                init_cash = func(init_cash_raw, group_lens, cash_sharing)
            else:
                group_lens = wrapper.grouper.get_group_lens()
                func = jit_reg.resolve_option(nb.init_cash_nb, jitted)
                init_cash = func(init_cash_raw, group_lens, cash_sharing, split_shared=split_shared)
        wrap_kwargs = merge_dicts(dict(name_or_index="init_cash"), wrap_kwargs)
        return wrapper.wrap_reduced(init_cash, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_cash(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        free: bool = False,
        cash_sharing: tp.Optional[bool] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash balance series per column/group.

        For `free`, see `Portfolio.get_cash_flow`."""
        if not isinstance(cls_or_self, type):
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    free=free,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(init_cash)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        "init_cash",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash = func(
                wrapper.shape_2d,
                to_2d_array(cash_flow),
                group_lens,
                to_1d_array(init_cash),
                cash_deposits_grouped=to_2d_array(cash_deposits),
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        "init_cash",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            func = jit_reg.resolve_option(nb.cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash = func(
                to_2d_array(cash_flow),
                to_1d_array(init_cash),
                cash_deposits=to_2d_array(cash_deposits),
            )
        return wrapper.wrap(cash, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Value ############# #

    @class_or_instancemethod
    def get_init_price(
        cls_or_self,
        init_price_raw: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial price per column."""
        if not isinstance(cls_or_self, type):
            if init_price_raw is None:
                init_price_raw = cls_or_self._init_price
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_price_raw)
            checks.assert_not_none(wrapper)

        init_price = np.broadcast_to(to_1d_array(init_price_raw), (wrapper.shape_2d[1],))
        wrap_kwargs = merge_dicts(dict(name_or_index="init_price"), wrap_kwargs)
        return wrapper.wrap_reduced(init_price, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_init_position_value(
        cls_or_self,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial position value per column."""
        if not isinstance(cls_or_self, type):
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if init_position is None:
                init_position = 0.0
            checks.assert_not_none(init_price)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.init_position_value_nb, jitted)
        init_position_value = func(
            n_cols=wrapper.shape_2d[1],
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position_value, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_init_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        init_cash: tp.Optional[tp.MaybeSeries] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial value per column/group.

        Includes initial cash and the value of initial position."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    "init_cash",
                    group_by=group_by,
                    split_shared=split_shared,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(init_cash)
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.init_value_grouped_nb, jitted)
            init_value = func(group_lens, to_1d_array(init_position_value), to_1d_array(init_cash))
        else:
            func = jit_reg.resolve_option(nb.init_value_nb, jitted)
            init_value = func(to_1d_array(init_position_value), to_1d_array(init_cash))
        wrap_kwargs = merge_dicts(dict(name_or_index="init_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_input_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_sharing: tp.Optional[bool] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total input value per column/group.

        Includes initial value and any cash deposited at any point in time."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits_raw is None:
                cash_deposits_raw = cls_or_self._cash_deposits
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(init_value)
            if cash_deposits_raw is None:
                cash_deposits_raw = 0.0
            checks.assert_not_none(wrapper)

        cash_deposits_raw = to_2d_array(cash_deposits_raw)
        cash_deposits_sum = cash_deposits_raw.sum(axis=0)
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.init_cash_grouped_nb, jitted)
            input_value = func(cash_deposits_sum, group_lens, cash_sharing)
        else:
            group_lens = wrapper.grouper.get_group_lens()
            func = jit_reg.resolve_option(nb.init_cash_nb, jitted)
            input_value = func(cash_deposits_sum, group_lens, cash_sharing, split_shared=split_shared)
        input_value += to_1d_array(init_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="input_value"), wrap_kwargs)
        return wrapper.wrap_reduced(input_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_value(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        assets: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset value series per column/group."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_value = func(to_2d_array(close), to_2d_array(assets))
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            asset_value = func(asset_value, group_lens)
        return wrapper.wrap(asset_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_gross_exposure(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get gross exposure."""
        direction = map_enum_fields(direction, Direction)

        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value)
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.gross_exposure_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        gross_exposure = func(to_2d_array(asset_value), to_2d_array(value))
        return wrapper.wrap(gross_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_net_exposure(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        long_exposure: tp.Optional[tp.SeriesFrame] = None,
        short_exposure: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get net exposure."""
        if not isinstance(cls_or_self, type):
            if long_exposure is None:
                long_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction=Direction.LongOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if short_exposure is None:
                short_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction=Direction.ShortOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(long_exposure)
            checks.assert_not_none(short_exposure)
            checks.assert_not_none(wrapper)

        net_exposure = to_2d_array(long_exposure) - to_2d_array(short_exposure)
        return wrapper.wrap(net_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash: tp.Optional[tp.SeriesFrame] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get portfolio value series per column/group.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with the initial cash balance and position being that of the
        entire group. Useful for generating returns and comparing assets within the same group."""
        if not isinstance(cls_or_self, type):
            if cash is None:
                cash = cls_or_self.resolve_shortcut_attr("cash", group_by=group_by, jitted=jitted, chunked=chunked)
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        value = func(to_2d_array(cash), to_2d_array(asset_value))
        return wrapper.wrap(value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_allocations(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get portfolio allocation series per column."""
        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    direction=direction,
                    group_by=False,
                    jitted=jitted,
                    chunked=chunked,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr("value", group_by=group_by, jitted=jitted, chunked=chunked)
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value)
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        if not wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Portfolio must be grouped. Provide group_by.")
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        func = jit_reg.resolve_option(nb.allocations_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        allocations = func(to_2d_array(asset_value), to_2d_array(value), group_lens)
        return wrapper.wrap(allocations, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_total_profit(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total profit per column/group.

        Calculated directly from order records (fast)."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if cash_earnings is None:
                cash_earnings = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if close is None:
                close = orders.close
            checks.assert_not_none(close)
            checks.assert_not_none(init_price)
            if init_position is None:
                init_position = 0.0
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        func = jit_reg.resolve_option(nb.total_profit_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        total_profit = func(
            wrapper.shape_2d,
            to_2d_array(close),
            orders.values,
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            cash_earnings=to_2d_array(cash_earnings),
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.total_profit_grouped_nb, jitted)
            total_profit = func(total_profit, group_lens)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_profit"), wrap_kwargs)
        return wrapper.wrap_reduced(total_profit, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_final_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total profit per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        final_value = to_1d_array(input_value) + to_1d_array(total_profit)
        wrap_kwargs = merge_dicts(dict(name_or_index="final_value"), wrap_kwargs)
        return wrapper.wrap_reduced(final_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_total_return(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total return per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        total_return = to_1d_array(total_profit) / to_1d_array(input_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get return series per column/group based on portfolio value."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr("value", group_by=group_by, jitted=jitted, chunked=chunked)
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        returns = func(
            to_2d_array(value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            log_returns=log_returns,
        )
        returns = wrapper.wrap(returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            returns = returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return returns

    @class_or_instancemethod
    def get_asset_pnl(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset (realized and unrealized) PnL series per column/group."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_pnl_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_pnl = func(
            to_1d_array(init_position_value),
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
        )
        return wrapper.wrap(asset_pnl, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_asset_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset return series per column/group.

        This type of returns is based solely on cash flows and asset value rather than portfolio
        value. It ignores passive cash and thus it will return the same numbers irrespective of the amount of
        cash currently available, even `np.inf`. The scale of returns is comparable to that of going
        all in and keeping available cash at zero."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_returns = func(
            to_1d_array(init_position_value),
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
            log_returns=log_returns,
        )
        asset_returns = wrapper.wrap(asset_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            asset_returns = asset_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return asset_returns

    @class_or_instancemethod
    def get_market_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get market value series per column/group.

        If grouped, evenly distributes the initial cash among assets in the group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.market_value_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                group_lens,
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            func = jit_reg.resolve_option(nb.market_value_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
            )
        return wrapper.wrap(market_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_market_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get market return series per column/group."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True,
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(market_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        market_returns = func(
            to_2d_array(market_value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            log_returns=log_returns,
        )
        market_returns = wrapper.wrap(market_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            market_returns = market_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return market_returns

    @class_or_instancemethod
    def get_bm_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark value series per column/group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_value`."""
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if isinstance(bm_close, bool):
                    if not bm_close:
                        return None
                    bm_close = None
                if bm_close is not None:
                    if cls_or_self.fillna_close:
                        bm_close = cls_or_self.filled_bm_close
        return cls_or_self.get_market_value(
            group_by=group_by,
            close=bm_close,
            init_value=init_value,
            cash_deposits=cash_deposits,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
        )

    @class_or_instancemethod
    def get_bm_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        bm_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark return series per column/group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_returns`."""
        if not isinstance(cls_or_self, type):
            bm_value = cls_or_self.resolve_shortcut_attr("bm_value", group_by=group_by, jitted=jitted, chunked=chunked)
            if bm_value is None:
                return None
        return cls_or_self.get_market_returns(
            group_by=group_by,
            init_value=init_value,
            cash_deposits=cash_deposits,
            market_value=bm_value,
            log_returns=log_returns,
            daily_returns=daily_returns,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
        )

    @class_or_instancemethod
    def get_total_market_return(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total market return."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(market_value)
            checks.assert_not_none(wrapper)

        input_value = to_1d_array(input_value)
        final_value = to_2d_array(market_value)[-1]
        total_return = (final_value - input_value) / input_value
        wrap_kwargs = merge_dicts(dict(name_or_index="total_market_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns_acc(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        returns: tp.Optional[tp.SeriesFrame] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        defaults: tp.KwargsLike = None,
        **kwargs,
    ) -> ReturnsAccessor:
        """Get returns accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`.

        !!! hint
            You can find most methods of this accessor as (cacheable) attributes of this portfolio."""
        if not isinstance(cls_or_self, type):
            if returns is None:
                if use_asset_returns:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "asset_returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                else:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
            if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
                bm_returns = cls_or_self.resolve_shortcut_attr(
                    "bm_returns",
                    log_returns=log_returns,
                    daily_returns=daily_returns,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            elif isinstance(bm_returns, bool) and not bm_returns:
                bm_returns = None
            if freq is None:
                freq = cls_or_self.wrapper.freq
        else:
            checks.assert_not_none(returns)

        if daily_returns:
            freq = "D"
        return returns.vbt.returns(
            bm_returns=bm_returns,
            log_returns=log_returns,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            **kwargs,
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """`Portfolio.get_returns_acc` with default arguments."""
        return self.get_returns_acc()

    @class_or_instancemethod
    def get_qs(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        returns: tp.Optional[tp.SeriesFrame] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        defaults: tp.KwargsLike = None,
        **kwargs,
    ) -> QSAdapterT:
        """Get quantstats adapter of type `vectorbtpro.returns.qs_adapter.QSAdapter`.

        `**kwargs` are passed to the adapter constructor."""
        from vectorbtpro.returns.qs_adapter import QSAdapter

        returns_acc = cls_or_self.get_returns_acc(
            group_by=group_by,
            returns=returns,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            jitted=jitted,
            chunked=chunked,
            defaults=defaults,
        )
        return QSAdapter(returns_acc, **kwargs)

    @property
    def qs(self) -> QSAdapterT:
        """`Portfolio.get_qs` with default arguments."""
        return self.get_qs()

    # ############# Resolution ############# #

    @property
    def self_aliases(self) -> tp.Set[str]:
        """Names to associate with this object."""
        return {"self", "portfolio", "pf"}

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Uses the following keys:

        * `use_asset_returns`: Whether to use `Portfolio.get_asset_returns` when resolving `returns` argument.
        * `trades_type`: Which trade type to use when resolving `trades` argument."""
        if "use_asset_returns" in final_kwargs:
            if attr == "returns" and final_kwargs["use_asset_returns"]:
                attr = "asset_returns"
        if "trades_type" in final_kwargs:
            trades_type = final_kwargs["trades_type"]
            if isinstance(final_kwargs["trades_type"], str):
                trades_type = map_enum_fields(trades_type, TradesType)
            if attr == "trades" and trades_type != self.trades_type:
                if trades_type == TradesType.EntryTrades:
                    attr = "entry_trades"
                elif trades_type == TradesType.ExitTrades:
                    attr = "exit_trades"
                else:
                    attr = "positions"
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process an object after resolution.

        Uses the following keys:

        * `incl_open`: Whether to include open trades/positions when resolving an argument
            that is an instance of `vectorbtpro.portfolio.trades.Trades`."""
        if "incl_open" in final_kwargs:
            if isinstance(out, Trades) and not final_kwargs["incl_open"]:
                out = out.status_closed
        return out

    def resolve_shortcut_attr(self, attr_name: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute that may have shortcut properties.

        If `attr_name` has a prefix `get_`, checks whether the respective shortcut property can be called.
        This way, complex call hierarchies can utilize cacheable properties."""
        if not attr_name.startswith("get_"):
            if "get_" + attr_name not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr_name), property):
                    return getattr(self, attr_name)
                return getattr(self, attr_name)(*args, **kwargs)
            attr_name = "get_" + attr_name

        if len(args) == 0:
            naked_attr_name = attr_name[4:]
            prop_name = naked_attr_name
            _kwargs = dict(kwargs)

            if "free" in _kwargs:
                if _kwargs.pop("free"):
                    prop_name = "free_" + naked_attr_name
            if "direction" in _kwargs:
                direction = map_enum_fields(_kwargs.pop("direction"), Direction)
                if direction == Direction.LongOnly:
                    prop_name = "longonly_" + naked_attr_name
                elif direction == Direction.ShortOnly:
                    prop_name = "shortonly_" + naked_attr_name

            if prop_name in self.cls_dir:
                prop = getattr(type(self), prop_name)
                options = getattr(prop, "options", {})

                can_call_prop = True
                if "group_by" in _kwargs:
                    group_by = _kwargs.pop("group_by")
                    group_aware = options.get("group_aware", True)
                    if group_aware:
                        if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                            can_call_prop = False
                    else:
                        group_by = _kwargs.pop("group_by")
                        if self.wrapper.grouper.is_grouping_enabled(group_by=group_by):
                            can_call_prop = False
                if can_call_prop:
                    _kwargs.pop("jitted", None)
                    _kwargs.pop("chunked", None)
                    for k, v in get_func_kwargs(getattr(type(self), attr_name)).items():
                        if k in _kwargs and v is not _kwargs.pop(k):
                            can_call_prop = False
                            break
                    if can_call_prop:
                        if len(_kwargs) > 0:
                            can_call_prop = False
                        if can_call_prop:
                            return getattr(self, prop_name)

        return getattr(self, attr_name)(*args, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.portfolio`."""
        from vectorbtpro._settings import settings

        returns_cfg = settings["returns"]
        portfolio_stats_cfg = settings["portfolio"]["stats"]

        return merge_dicts(
            Analyzable.stats_defaults.__get__(self),
            dict(settings=dict(year_freq=returns_cfg["year_freq"], trades_type=self.trades_type)),
            portfolio_stats_cfg,
        )

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(title="Start", calc_func=lambda self: self.wrapper.index[0], agg_func=None, tags="wrapper"),
            end=dict(title="End", calc_func=lambda self: self.wrapper.index[-1], agg_func=None, tags="wrapper"),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            start_value=dict(title="Start Value", calc_func="init_value", tags="portfolio"),
            min_value=dict(title="Min Value", calc_func="value.vbt.min", tags="portfolio"),
            max_value=dict(title="Max Value", calc_func="value.vbt.max", tags="portfolio"),
            end_value=dict(title="End Value", calc_func="final_value", tags="portfolio"),
            cash_deposits=dict(
                title="Cash Deposits", calc_func="cash_deposits.vbt.sum", check_has_cash_deposits=True, tags="portfolio"
            ),
            cash_earnings=dict(
                title="Cash Earnings", calc_func="cash_earnings.vbt.sum", check_has_cash_earnings=True, tags="portfolio"
            ),
            total_return=dict(
                title="Total Return [%]",
                calc_func="total_return",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            bm_return=dict(
                title="Benchmark Return [%]",
                calc_func="bm_returns.vbt.returns.total",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_bm_returns=True,
                tags="portfolio",
            ),
            total_time_exposure=dict(
                title="Total Time Exposure [%]",
                calc_func="position_mask.vbt.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_gross_exposure=dict(
                title="Max Gross Exposure [%]",
                calc_func="gross_exposure.vbt.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_dd=dict(
                title="Max Drawdown [%]",
                calc_func="drawdowns.max_drawdown",
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=["portfolio", "drawdowns"],
            ),
            max_dd_duration=dict(
                title="Max Drawdown Duration",
                calc_func="drawdowns.max_duration",
                fill_wrap_kwargs=True,
                tags=["portfolio", "drawdowns", "duration"],
            ),
            total_orders=dict(
                title="Total Orders",
                calc_func="orders.count",
                tags=["portfolio", "orders"],
            ),
            total_fees_paid=dict(
                title="Total Fees Paid",
                calc_func="orders.fees.sum",
                tags=["portfolio", "orders"],
            ),
            total_trades=dict(
                title="Total Trades",
                calc_func="trades.count",
                incl_open=True,
                tags=["portfolio", "trades"],
            ),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="trades.win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func="trades.returns.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func="trades.returns.min",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func="trades.winning.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func="trades.losing.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func="trades.winning.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func="trades.losing.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func="trades.profit_factor",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func="trades.expectancy",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            sharpe_ratio=dict(
                title="Sharpe Ratio",
                calc_func="returns_acc.sharpe_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            calmar_ratio=dict(
                title="Calmar Ratio",
                calc_func="returns_acc.calmar_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            omega_ratio=dict(
                title="Omega Ratio",
                calc_func="returns_acc.omega_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            sortino_ratio=dict(
                title="Sortino Ratio",
                calc_func="returns_acc.sortino_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    def returns_stats(
        self,
        group_by: tp.GroupByLike = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        defaults: tp.KwargsLike = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Compute various statistics on returns of this portfolio.

        See `Portfolio.returns_acc` and `vectorbtpro.returns.accessors.ReturnsAccessor.metrics`.

        `kwargs` will be passed to `vectorbtpro.returns.accessors.ReturnsAccessor.stats` method.
        If `bm_returns` is not set, uses `Portfolio.get_market_returns`."""
        returns_acc = self.get_returns_acc(
            group_by=group_by,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            defaults=defaults,
            chunked=chunked,
        )
        return getattr(returns_acc, "stats")(**kwargs)

    # ############# Plotting ############# #

    def plot_trade_signals(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_positions: tp.Union[bool, str] = "zones",
        long_entry_trace_kwargs: tp.KwargsLike = None,
        short_entry_trace_kwargs: tp.KwargsLike = None,
        long_exit_trace_kwargs: tp.KwargsLike = None,
        short_exit_trace_kwargs: tp.KwargsLike = None,
        long_shape_kwargs: tp.KwargsLike = None,
        short_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of trade signals.

        Markers and shapes are colored by trade direction (green = long, red = short)."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        entry_trades = self.resolve_shortcut_attr("entry_trades")
        exit_trades = self.resolve_shortcut_attr("exit_trades")
        positions = self.resolve_shortcut_attr("positions")

        fig = entry_trades.plot_signals(
            column=column,
            long_entry_trace_kwargs=long_entry_trace_kwargs,
            short_entry_trace_kwargs=short_entry_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **kwargs,
        )
        fig = exit_trades.plot_signals(
            column=column,
            plot_ohlc=False,
            plot_close=False,
            long_exit_trace_kwargs=long_exit_trace_kwargs,
            short_exit_trace_kwargs=short_exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if isinstance(plot_positions, bool):
            if plot_positions:
                plot_positions = "zones"
            else:
                plot_positions = None
        if plot_positions is not None:
            if plot_positions.lower() == "zones":
                long_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["green"]),
                    long_shape_kwargs,
                )
                short_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["red"]),
                    short_shape_kwargs,
                )
            elif plot_positions.lower() == "lines":
                base_shape_kwargs = dict(
                    type="line",
                    line=dict(dash="dot"),
                    xref=Rep("xref"),
                    yref=Rep("yref"),
                    x0=Rep("start_index"),
                    x1=Rep("end_index"),
                    y0=RepFunc(lambda record: record["entry_price"]),
                    y1=RepFunc(lambda record: record["exit_price"]),
                    opacity=0.75,
                )
                long_shape_kwargs = atomic_dict(merge_dicts(
                    base_shape_kwargs,
                    dict(line=dict(color=plotting_cfg["contrast_color_schema"]["green"])),
                    long_shape_kwargs,
                ))
                short_shape_kwargs = atomic_dict(merge_dicts(
                    base_shape_kwargs,
                    dict(line=dict(color=plotting_cfg["contrast_color_schema"]["red"])),
                    short_shape_kwargs,
                ))
            else:
                raise ValueError(f"Invalid option plot_positions='{plot_positions}'")
            fig = positions.direction_long.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=long_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x",
                yref=fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y",
                fig=fig,
            )
            fig = positions.direction_short.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=short_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x",
                yref=fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y",
                fig=fig,
            )
        return fig

    def plot_asset_flow(
        self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of asset flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        asset_flow = self.resolve_shortcut_attr("asset_flow", direction=direction, jitted=jitted, chunked=chunked)
        asset_flow = self.select_col_from_obj(asset_flow, column, wrapper=self.wrapper.regroup(False))
        kwargs = merge_dicts(
            dict(trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["brown"]), name="Assets")),
            kwargs,
        )
        fig = asset_flow.vbt.lineplot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_cash_flow(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        free: bool = False,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cash flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        cash_flow = self.resolve_shortcut_attr(
            "cash_flow",
            group_by=group_by,
            free=free,
            jitted=jitted,
            chunked=chunked,
        )
        cash_flow = self.select_col_from_obj(cash_flow, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["green"]), name="Cash")),
            kwargs,
        )
        fig = cash_flow.vbt.lineplot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_assets(
        self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of assets.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        assets = self.resolve_shortcut_attr("assets", direction=direction, jitted=jitted, chunked=chunked)
        assets = self.select_col_from_obj(assets, column, wrapper=self.wrapper.regroup(False))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["brown"]), name="Assets"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["brown"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = assets.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_cash(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        free: bool = False,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cash balance.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        init_cash = self.resolve_shortcut_attr("init_cash", group_by=group_by, jitted=jitted, chunked=chunked)
        init_cash = self.select_col_from_obj(init_cash, column, wrapper=self.wrapper.regroup(group_by))
        cash = self.resolve_shortcut_attr("cash", group_by=group_by, free=free, jitted=jitted, chunked=chunked)
        cash = self.select_col_from_obj(cash, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["green"]), name="Cash"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["green"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = cash.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=init_cash,
                    x1=x_domain[1],
                    y1=init_cash,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_asset_value(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of asset value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        asset_value = self.resolve_shortcut_attr(
            "asset_value",
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        asset_value = self.select_col_from_obj(asset_value, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["cyan"]), name="Asset Value"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = asset_value.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_value(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        init_cash = self.resolve_shortcut_attr("init_cash", group_by=group_by, jitted=jitted, chunked=chunked)
        init_cash = self.select_col_from_obj(init_cash, column, wrapper=self.wrapper.regroup(group_by))
        value = self.resolve_shortcut_attr("value", group_by=group_by, jitted=jitted, chunked=chunked)
        value = self.select_col_from_obj(value, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value"),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = value.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=init_cash,
                    x1=x_domain[1],
                    y1=init_cash,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_allocations(
        self,
        column: tp.Optional[tp.Label] = None,
        line_shape: str = "hv",
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one group of allocations."""
        filled_allocations = self.resolve_shortcut_attr(
            "allocations",
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        filled_allocations = self.select_col_from_obj(filled_allocations, column, obj_ungrouped=True)
        group_names = self.wrapper.grouper.get_index().names
        filled_allocations = filled_allocations.vbt.drop_levels(group_names, strict=False)
        fig = filled_allocations.vbt.areaplot(line_shape=line_shape, **kwargs)
        return fig

    def plot_cum_returns(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        log_returns: bool = False,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cumulative returns.

        If `bm_returns` is None, will use `Portfolio.get_market_returns`.

        `**kwargs` are passed to `vectorbtpro.returns.accessors.ReturnsSRAccessor.plot_cumulative`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
            bm_returns = self.resolve_shortcut_attr(
                "bm_returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        elif isinstance(bm_returns, bool) and not bm_returns:
            bm_returns = None
        else:
            bm_returns = broadcast_to(bm_returns, self.obj)
        bm_returns = self.select_col_from_obj(bm_returns, column, wrapper=self.wrapper.regroup(group_by))
        if use_asset_returns:
            returns = self.resolve_shortcut_attr(
                "asset_returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        else:
            returns = self.resolve_shortcut_attr(
                "returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        returns = self.select_col_from_obj(returns, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                bm_returns=bm_returns,
                main_kwargs=dict(
                    trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value"),
                ),
                hline_shape_kwargs=dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
            ),
            kwargs,
        )
        return returns.vbt.returns(log_returns=log_returns).plot_cumulative(**kwargs)

    def plot_drawdowns(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of drawdowns.

        `**kwargs` are passed to `vectorbtpro.generic.drawdowns.Drawdowns.plot`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        kwargs = merge_dicts(
            dict(close_trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value")),
            kwargs,
        )
        return self.resolve_shortcut_attr("drawdowns", group_by=group_by).plot(column=column, **kwargs)

    def plot_underwater(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of underwater.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        drawdown = self.resolve_shortcut_attr("drawdown", group_by=group_by, jitted=jitted, chunked=chunked)
        drawdown = self.select_col_from_obj(drawdown, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(color=plotting_cfg["color_schema"]["red"]),
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3),
                    fill="tozeroy",
                    name="Drawdown",
                )
            ),
            kwargs,
        )
        fig = drawdown.vbt.lineplot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        yaxis = "yaxis" + yref[1:]
        fig.layout[yaxis]["tickformat"] = "%"
        return fig

    def plot_gross_exposure(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of gross exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        gross_exposure = self.resolve_shortcut_attr(
            "gross_exposure",
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        gross_exposure = self.select_col_from_obj(gross_exposure, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["pink"]), name="Exposure"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = gross_exposure.vbt.plot_against(1, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=1,
                    x1=x_domain[1],
                    y1=1,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_net_exposure(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of net exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        net_exposure = self.resolve_shortcut_attr("net_exposure", group_by=group_by, jitted=jitted, chunked=chunked)
        net_exposure = self.select_col_from_obj(net_exposure, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["pink"]), name="Exposure"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = net_exposure.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.plot`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.portfolio`."""
        from vectorbtpro._settings import settings

        returns_cfg = settings["returns"]
        portfolio_plots_cfg = settings["portfolio"]["plots"]

        return merge_dicts(
            Analyzable.plots_defaults.__get__(self),
            dict(settings=dict(year_freq=returns_cfg["year_freq"], trades_type=self.trades_type)),
            portfolio_plots_cfg,
        )

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            orders=dict(
                title="Orders",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="orders.plot",
                tags=["portfolio", "orders"],
            ),
            trades=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="trades.plot",
                tags=["portfolio", "trades"],
            ),
            trade_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func="trades.plot_pnl",
                tags=["portfolio", "trades"],
            ),
            trade_signals=dict(
                title="Trade Signals",
                yaxis_kwargs=dict(title="Trade Signals"),
                check_is_not_grouped=True,
                plot_func="plot_trade_signals",
                tags=["portfolio", "trades"],
            ),
            asset_flow=dict(
                title="Asset Flow",
                yaxis_kwargs=dict(title="Asset flow"),
                check_is_not_grouped=True,
                plot_func="plot_asset_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            cash_flow=dict(
                title="Cash Flow",
                yaxis_kwargs=dict(title="Cash flow"),
                plot_func="plot_cash_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            assets=dict(
                title="Assets",
                yaxis_kwargs=dict(title="Assets"),
                check_is_not_grouped=True,
                plot_func="plot_assets",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            cash=dict(
                title="Cash",
                yaxis_kwargs=dict(title="Cash"),
                plot_func="plot_cash",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            asset_value=dict(
                title="Asset Value",
                yaxis_kwargs=dict(title="Asset value"),
                plot_func="plot_asset_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets", "value"],
            ),
            value=dict(
                title="Value",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value"],
            ),
            cum_returns=dict(
                title="Cumulative Returns",
                yaxis_kwargs=dict(title="Cumulative returns"),
                plot_func="plot_cum_returns",
                pass_hline_shape_kwargs=True,
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "returns"],
            ),
            drawdowns=dict(
                title="Drawdowns",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_drawdowns",
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            underwater=dict(
                title="Underwater",
                yaxis_kwargs=dict(title="Drawdown"),
                plot_func="plot_underwater",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            gross_exposure=dict(
                title="Gross Exposure",
                yaxis_kwargs=dict(title="Gross exposure"),
                plot_func="plot_gross_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
            net_exposure=dict(
                title="Net Exposure",
                yaxis_kwargs=dict(title="Net exposure"),
                plot_func="plot_net_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
        )
    )

    plot = Analyzable.plots

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_in_output_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build in-output config documentation."""
        if source_cls is None:
            source_cls = Portfolio
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "in_output_config").__doc__)).substitute(
            {"in_output_config": cls.in_output_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_in_output_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Data.in_output_config`."""
        __pdoc__[cls.__name__ + ".in_output_config"] = cls.build_in_output_config_doc(source_cls=source_cls)


Portfolio.override_in_output_config_doc(__pdoc__)
Portfolio.override_metrics_doc(__pdoc__)
Portfolio.override_subplots_doc(__pdoc__)

__pdoc__["Portfolio.plot"] = "See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."
