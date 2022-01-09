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
>>> from vectorbtpro.base.reshaping import broadcast, to_2d_array
>>> from vectorbtpro.base.indexing import flex_select_auto_nb
>>> from vectorbtpro.portfolio.enums import SizeType, Direction, NoOrder, OrderStatus, OrderSide
>>> from vectorbtpro.portfolio import nb
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
...     return nb.order_nb(
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
>>> pf.value.vbt.plot()
```

![](/assets/images/portfolio_value.svg)

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
    otherwise on single columns. You can pass wrapper arguments with `wrapper_kwargs`.

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
...     return nb.order_nb(
...         size=nb.get_elem_nb(c, size),
...         price=nb.get_elem_nb(c, price),
...         fees=nb.get_elem_nb(c, fees),
...         slippage=nb.get_elem_nb(c, slippage),
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
...     np.asarray(size),
...     np.asarray(ohlcv['Open']),
...     np.asarray(0.001),
...     np.asarray(0.001),
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
add the suffix '_pcgs' to the name of the array (see `Portfolio.in_outputs_indexing_func` on supported suffixes).

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
Name: agg_func_mean, dtype: object
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

Here's an example of a parametrized metric. Let's get the number of trades with PnL over some amount:

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
>>> pf.plot(column=10)
```

![](/assets/images/portfolio_col_plot.svg)

To plot a single column of a grouped portfolio:

```pycon
>>> pf_grouped = vbt.Portfolio.from_random_signals(
...     close, n=[10, 20, 30, 40], seed=42, freq='d',
...     group_by=['group1', 'group1', 'group2', 'group2'])

>>> pf_grouped.plot(column=10, group_by=False)
```

To plot a single group of a grouped portfolio:

```pycon
>>> pf_grouped.plot(column='group1')
UserWarning: Subplot 'orders' does not support grouped data
UserWarning: Subplot 'trade_pnl' does not support grouped data
```

![](/assets/images/portfolio_group_plot.svg)

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
... )
```

![](/assets/images/portfolio_plot_drawdowns.svg)

### Custom subplots

To create a new subplot, a preferred way is to pass a plotting function:

```pycon
>>> def plot_order_size(pf, size, column=None, add_trace_kwargs=None, fig=None):
...     size = pf.select_one_from_obj(size, pf.wrapper.regroup(False), column=column)
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
... )
```

![](/assets/images/portfolio_plot_custom.svg)

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
...             template_mapping=dict(
...                 window=10
...             )
...         )
...     )
... ).show_svg()
```

You can also replace templates across all subplots by using the global template mapping:

```pycon
>>> pf.plot(subplots, column=10, template_mapping=dict(window=10))
```

![](/assets/images/portfolio_plot_path.svg)
"""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array, broadcast, broadcast_to, to_pd_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.stats_builder import StatsBuilderMixin
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import require_call_seq, build_call_seq
from vectorbtpro.portfolio.decorators import attach_shortcut_properties, attach_returns_acc_methods
from vectorbtpro.portfolio.enums import *
from vectorbtpro.portfolio.logs import Logs
from vectorbtpro.portfolio.orders import Orders
from vectorbtpro.portfolio.trades import Trades, EntryTrades, ExitTrades, Positions
from vectorbtpro.records import nb as records_nb
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.signals.generators import RANDNX, RPROBNX
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.decorators import custom_property, cached_property, class_or_instancemethod
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.mapping import to_mapping
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import Rep, RepEval, RepFunc, deep_substitute

try:
    import quantstats as qs
except ImportError:
    QSAdapterT = tp.Any
else:
    from vectorbtpro.returns.qs_adapter import QSAdapter as QSAdapterT

__pdoc__ = {}


def fix_wrapper_for_records(pf: "Portfolio") -> ArrayWrapper:
    """Allow flags for records that were restricted for portfolio."""
    if pf.cash_sharing:
        return pf.wrapper.replace(allow_enable=True, allow_modify=True)
    return pf.wrapper


returns_acc_config = ReadonlyConfig(
    {
        'daily_returns': dict(
            source_name='daily'
        ),
        'annual_returns': dict(
            source_name='annual'
        ),
        'cumulative_returns': dict(
            source_name='cumulative'
        ),
        'annualized_return': dict(
            source_name='annualized'
        ),
        'annualized_volatility': dict(),
        'calmar_ratio': dict(),
        'omega_ratio': dict(),
        'sharpe_ratio': dict(),
        'deflated_sharpe_ratio': dict(),
        'downside_risk': dict(),
        'sortino_ratio': dict(),
        'information_ratio': dict(),
        'beta': dict(),
        'alpha': dict(),
        'tail_ratio': dict(),
        'value_at_risk': dict(),
        'cond_value_at_risk': dict(),
        'capture': dict(),
        'up_capture': dict(),
        'down_capture': dict(),
        'drawdown': dict(),
        'max_drawdown': dict()
    }
)
"""_"""

__pdoc__['returns_acc_config'] = f"""Config of returns accessor methods to be attached to `Portfolio`.

```python
{returns_acc_config.prettify()}
```
"""

shortcut_config = ReadonlyConfig(
    {
        'filled_close': dict(
            group_by_aware=False,
            decorator=cached_property
        ),
        'filled_bm_close': dict(
            group_by_aware=False,
            decorator=cached_property
        ),
        'orders': dict(
            obj_type='records',
            field_aliases=('order_records',),
            wrap_func=lambda pf, obj, **kwargs: Orders(fix_wrapper_for_records(pf), obj, pf.close),
        ),
        'logs': dict(
            obj_type='records',
            field_aliases=('log_records',),
            wrap_func=lambda pf, obj, **kwargs: Logs(fix_wrapper_for_records(pf), obj)
        ),
        'entry_trades': dict(
            obj_type='records',
            field_aliases=('entry_trade_records',),
            wrap_func=lambda pf, obj, **kwargs: EntryTrades.from_records(pf.orders.wrapper, obj, pf.close)
        ),
        'exit_trades': dict(
            obj_type='records',
            field_aliases=('exit_trade_records',),
            wrap_func=lambda pf, obj, **kwargs: ExitTrades.from_records(pf.orders.wrapper, obj, pf.close)
        ),
        'positions': dict(
            obj_type='records',
            field_aliases=('position_records',),
            wrap_func=lambda pf, obj, **kwargs: Positions.from_records(pf.orders.wrapper, obj, pf.close)
        ),
        'trades': dict(
            obj_type='records',
            field_aliases=('trade_records',),
            wrap_func=lambda pf, obj, **kwargs: Trades.from_records(pf.orders.wrapper, obj, pf.close)
        ),
        'drawdowns': dict(
            obj_type='records',
            field_aliases=('drawdown_records',),
            wrap_func=lambda pf, obj, **kwargs: Drawdowns.from_records(pf.orders.wrapper.regroup(False), obj, pf.close)
        ),
        'init_position': dict(
            obj_type='red_array',
            group_by_aware=False
        ),
        'asset_flow': dict(
            group_by_aware=False
        ),
        'longonly_asset_flow': dict(
            method_name='get_asset_flow',
            group_by_aware=False,
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_asset_flow': dict(
            method_name='get_asset_flow',
            group_by_aware=False,
            method_kwargs=dict(direction='shortonly')
        ),
        'assets': dict(
            group_by_aware=False
        ),
        'longonly_assets': dict(
            method_name='get_assets',
            group_by_aware=False,
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_assets': dict(
            method_name='get_assets',
            group_by_aware=False,
            method_kwargs=dict(direction='shortonly')
        ),
        'position_mask': dict(),
        'longonly_position_mask': dict(
            method_name='get_position_mask',
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_position_mask': dict(
            method_name='get_position_mask',
            method_kwargs=dict(direction='shortonly')
        ),
        'position_coverage': dict(
            obj_type='red_array'
        ),
        'longonly_position_coverage': dict(
            method_name='get_position_coverage',
            obj_type='red_array',
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_position_coverage': dict(
            method_name='get_position_coverage',
            obj_type='red_array',
            method_kwargs=dict(direction='shortonly')
        ),
        'init_cash': dict(
            obj_type='red_array'
        ),
        'cash_deposits': dict(),
        'cash_earnings': dict(),
        'cash_flow': dict(),
        'free_cash_flow': dict(
            method_name='get_cash_flow',
            method_kwargs=dict(free=True)
        ),
        'cash': dict(),
        'free_cash': dict(
            method_name='get_cash',
            method_kwargs=dict(free=True)
        ),
        'init_position_value': dict(
            obj_type='red_array',
            group_by_aware=False
        ),
        'init_value': dict(
            obj_type='red_array'
        ),
        'input_value': dict(
            obj_type='red_array'
        ),
        'asset_value': dict(),
        'longonly_asset_value': dict(
            method_name='get_asset_value',
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_asset_value': dict(
            method_name='get_asset_value',
            method_kwargs=dict(direction='shortonly')
        ),
        'gross_exposure': dict(),
        'longonly_gross_exposure': dict(
            method_name='get_gross_exposure',
            method_kwargs=dict(direction='longonly')
        ),
        'shortonly_gross_exposure': dict(
            method_name='get_gross_exposure',
            method_kwargs=dict(direction='shortonly')
        ),
        'net_exposure': dict(),
        'value': dict(),
        'total_profit': dict(
            obj_type='red_array'
        ),
        'final_value': dict(
            obj_type='red_array'
        ),
        'total_return': dict(
            obj_type='red_array'
        ),
        'returns': dict(),
        'asset_returns': dict(),
        'market_value': dict(),
        'market_returns': dict(),
        'bm_value': dict(),
        'bm_returns': dict(),
        'total_market_return': dict(
            obj_type='red_array'
        ),
        'daily_returns': dict(),
        'annual_returns': dict(),
        'cumulative_returns': dict(),
        'annualized_return': dict(
            obj_type='red_array'
        ),
        'annualized_volatility': dict(
            obj_type='red_array'
        ),
        'calmar_ratio': dict(
            obj_type='red_array'
        ),
        'omega_ratio': dict(
            obj_type='red_array'
        ),
        'sharpe_ratio': dict(
            obj_type='red_array'
        ),
        'deflated_sharpe_ratio': dict(
            obj_type='red_array'
        ),
        'downside_risk': dict(
            obj_type='red_array'
        ),
        'sortino_ratio': dict(
            obj_type='red_array'
        ),
        'information_ratio': dict(
            obj_type='red_array'
        ),
        'beta': dict(
            obj_type='red_array'
        ),
        'alpha': dict(
            obj_type='red_array'
        ),
        'tail_ratio': dict(
            obj_type='red_array'
        ),
        'value_at_risk': dict(
            obj_type='red_array'
        ),
        'cond_value_at_risk': dict(
            obj_type='red_array'
        ),
        'capture': dict(
            obj_type='red_array'
        ),
        'up_capture': dict(
            obj_type='red_array'
        ),
        'down_capture': dict(
            obj_type='red_array'
        ),
        'drawdown': dict(),
        'max_drawdown': dict(
            obj_type='red_array'
        )
    }
)
"""_"""

__pdoc__['shortcut_config'] = f"""Config of shortcut properties to be attached to `Portfolio`.

```python
{shortcut_config.prettify()}
```
"""

PortfolioT = tp.TypeVar("PortfolioT", bound="Portfolio")


class MetaPortfolio(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    pass


@attach_shortcut_properties(shortcut_config)
@attach_returns_acc_methods(returns_acc_config)
class Portfolio(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaPortfolio):
    """Class for modeling portfolio and measuring its performance.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        close (array_like): Last asset price at each time step.
        order_records (array_like): A structured NumPy array of order records.
        log_records (array_like): A structured NumPy array of log records.
        cash_sharing (bool): Whether to share cash within the same group.
        init_cash (InitCashMode or array_like of float): Initial capital.

            Can be provided in a format suitable for flexible indexing.
        init_position (array_like of float): Initial position.

            Can be provided in a format suitable for flexible indexing.
        cash_deposits (array_like of float): Cash deposited/withdrawn at each timestamp.

            Can be provided in a format suitable for flexible indexing with `flex_2d=False`
            (that is, 1-dim array means an element per row rather than column).
        cash_earnings (array_like of float): Earnings added at each timestamp.

            Can be provided in a format suitable for flexible indexing with `flex_2d=False`
            (that is, 1-dim array means an element per row rather than column).
        call_seq (array_like of int): Sequence of calls per row and group. Defaults to None.
        in_outputs (namedtuple): Named tuple with in-output objects.

            To substitute `Portfolio` attributes, provide already broadcasted and grouped objects.
            Also see `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.
        use_in_outputs (bool): Whether to return in-output objects when calling properties.
        fillna_close (bool): Whether to forward and backward fill NaN values in `close`.

            Applied after the simulation to avoid NaNs in asset value.

            See `Portfolio.get_filled_close`.
        trades_type (str or int): Default `vectorbtpro.portfolio.trades.Trades` to use across `Portfolio`.

            See `vectorbtpro.portfolio.enums.TradesType`.

    For defaults, see `vectorbtpro._settings.portfolio`.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Portfolio.replace`."""

    def __init__(self,
                 wrapper: ArrayWrapper,
                 close: tp.ArrayLike,
                 order_records: tp.RecordArray,
                 log_records: tp.RecordArray,
                 cash_sharing: bool,
                 init_cash: tp.Union[int, tp.FlexArray],
                 init_position: tp.FlexArray = np.asarray(0.),
                 cash_deposits: tp.FlexArray = np.asarray(0.),
                 cash_earnings: tp.FlexArray = np.asarray(0.),
                 call_seq: tp.Optional[tp.Array2d] = None,
                 in_outputs: tp.Optional[tp.NamedTuple] = None,
                 use_in_outputs: tp.Optional[bool] = None,
                 bm_close: tp.Optional[tp.ArrayLike] = None,
                 fillna_close: tp.Optional[bool] = None,
                 trades_type: tp.Optional[tp.Union[int, str]] = None,
                 **kwargs) -> None:

        from vectorbtpro._settings import settings
        portfolio_cfg = settings['portfolio']

        if use_in_outputs is None:
            use_in_outputs = portfolio_cfg['use_in_outputs']
        if fillna_close is None:
            fillna_close = portfolio_cfg['fillna_close']
        if trades_type is None:
            trades_type = portfolio_cfg['trades_type']
        if isinstance(trades_type, str):
            trades_type = map_enum_fields(trades_type, TradesType)
        if cash_sharing:
            if wrapper.grouper.allow_enable or wrapper.grouper.allow_modify:
                wrapper = wrapper.replace(allow_enable=False, allow_modify=False)

        Wrapping.__init__(
            self,
            wrapper,
            close=close,
            order_records=order_records,
            log_records=log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            call_seq=call_seq,
            in_outputs=in_outputs,
            use_in_outputs=use_in_outputs,
            bm_close=bm_close,
            fillna_close=fillna_close,
            trades_type=trades_type,
            **kwargs
        )
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)

        self._close = close
        self._order_records = order_records
        self._log_records = log_records
        self._cash_sharing = cash_sharing
        self._init_cash = init_cash
        self._init_position = init_position
        self._cash_deposits = cash_deposits
        self._cash_earnings = cash_earnings
        self._call_seq = call_seq
        self._in_outputs = in_outputs
        self._use_in_outputs = use_in_outputs
        self._bm_close = bm_close
        self._fillna_close = fillna_close
        self._trades_type = trades_type

    # ############# In-outputs ############# #

    def get_in_output(self,
                      field: str,
                      force_wrapping: bool = False,
                      group_by: tp.GroupByLike = None,
                      wrapper: tp.Optional[ArrayWrapper] = None,
                      wrap_kwargs: tp.KwargsLike = None,
                      wrap_func: tp.Optional[tp.Callable] = None,
                      **kwargs) -> tp.Optional[tp.AnyArray]:
        """Get wrapped in-output.

        See `Portfolio.in_outputs_indexing_func` for parsing rules.

        `**kwargs` are passed together with `self`, the object, `group_by`, and `wrap_kwargs` to `wrap_func`."""
        if wrapper is None:
            wrapper = self.wrapper
        if wrap_kwargs is None:
            wrap_kwargs = {}
        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)

        def _find_field(field_aliases: tp.Iterable[str], group_by_aware: bool = True) -> tp.Optional[str]:
            fields = set(self.in_outputs._fields)
            for field in field_aliases:
                if field in fields:
                    return field
                if is_grouped:
                    if group_by_aware:
                        if field + '_pg' in fields:
                            return field + '_pg'
                        if field + '_pcg' in fields:
                            return field + '_pcg'
                        if self.cash_sharing:
                            if field + '_pcgs' in fields:
                                return field + '_pcgs'
                    else:
                        if field + '_pc' in fields:
                            return field + '_pc'
                        if not self.cash_sharing:
                            if field + '_pcgs' in fields:
                                return field + '_pcgs'
                else:
                    if field + '_pc' in fields:
                        return field + '_pc'
                    if field + '_pcg' in fields:
                        return field + '_pcg'
                    if field + '_pcgs' in fields:
                        return field + '_pcgs'
            return None

        def _wrap_1d_grouped(obj: tp.Array, name_or_index: str) -> tp.Series:
            _wrap_kwargs = merge_dicts(dict(name_or_index=name_or_index), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=group_by, **_wrap_kwargs)

        def _wrap_1d(obj: tp.Array, name_or_index: str) -> tp.Series:
            _wrap_kwargs = merge_dicts(dict(name_or_index=name_or_index), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=False, **_wrap_kwargs)

        def _wrap_2d_grouped(obj: tp.Array) -> tp.Frame:
            return wrapper.wrap(obj, group_by=group_by, **wrap_kwargs)

        def _wrap_2d(obj: tp.Array) -> tp.Frame:
            return wrapper.wrap(obj, group_by=False, **wrap_kwargs)

        if field in self.cls_dir:
            method_or_prop = getattr(type(self), field)
            options = getattr(method_or_prop, 'options', {})
            obj_type = options.get('obj_type', None)
            group_by_aware = options.get('group_by_aware', None)
            field_aliases = options.get('field_aliases', None)
            if field_aliases is None:
                field_aliases = []
            field_aliases = [field, *field_aliases]
            if obj_type is None:
                raise TypeError(f"Cannot parse in-output '{field}': option 'obj_type' is missing")
            if group_by_aware is None:
                raise TypeError(f"Cannot parse in-output '{field}': option 'group_by_aware' is missing")
            if obj_type == 'array':
                field_aliases = [*[alias + '_2d' for alias in field_aliases if '_2d' not in alias], *field_aliases]
            elif obj_type == 'red_array':
                field_aliases = [*[alias + '_1d' for alias in field_aliases if '_1d' not in alias], *field_aliases]

            found_field = _find_field(field_aliases, group_by_aware=group_by_aware)
            if found_field is None:
                raise AttributeError(f"Field '{field}' not found in in_outputs")
            obj = getattr(self.in_outputs, found_field)
            if obj is None or isinstance(obj, bool):
                return obj
            if wrap_func is not None:
                return wrap_func(
                    self,
                    obj,
                    field=field,
                    force_wrapping=force_wrapping,
                    group_by=group_by,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs
                )
            if obj_type == 'array':
                if group_by_aware:
                    return _wrap_2d_grouped(obj)
                return _wrap_2d(obj)
            elif obj_type == 'red_array':
                if group_by_aware:
                    return _wrap_1d_grouped(obj, field)
                return _wrap_1d(obj, field)
            if force_wrapping:
                raise NotImplementedError(f"Cannot wrap field '{found_field}'")
            return obj

        found_field = _find_field([field])
        if found_field is None:
            raise AttributeError(f"Field '{field}' not found in in_outputs")
        obj = getattr(self.in_outputs, found_field)
        if obj is None or isinstance(obj, bool):
            return obj
        if wrap_func is not None:
            return wrap_func(
                self,
                obj,
                field=field,
                force_wrapping=force_wrapping,
                group_by=group_by,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )
        if found_field.endswith('_pcgs'):
            if '_2d' in field:
                if is_grouped and self.cash_sharing:
                    return _wrap_2d_grouped(obj)
                return _wrap_2d(obj)
            if '_1d' in field:
                if is_grouped and self.cash_sharing:
                    return _wrap_1d_grouped(obj, field.replace('_pcgs', '').replace('_1d', ''))
                return _wrap_1d(obj, field.replace('_pcgs', '').replace('_1d', ''))
            if obj.ndim == 2:
                if is_grouped and self.cash_sharing:
                    return _wrap_2d_grouped(obj)
                return _wrap_2d(obj)
            if obj.ndim == 1:
                if is_grouped and self.cash_sharing:
                    return _wrap_1d_grouped(obj, field.replace('_pcgs', ''))
                return _wrap_1d(obj, field.replace('_pcgs', ''))
        if found_field.endswith('_pcg'):
            if '_2d' in field:
                if is_grouped:
                    return _wrap_2d_grouped(obj)
                return _wrap_2d(obj)
            if '_1d' in field:
                if is_grouped:
                    return _wrap_1d_grouped(obj, field.replace('_pcg', '').replace('_1d', ''))
                return _wrap_1d(obj, field.replace('_pcg', '').replace('_1d', ''))
            if obj.ndim == 2:
                if is_grouped:
                    return _wrap_2d_grouped(obj)
                return _wrap_2d(obj)
            if obj.ndim == 1:
                if is_grouped:
                    return _wrap_1d_grouped(obj, field.replace('_pcg', ''))
                return _wrap_1d(obj, field.replace('_pcg', ''))
        if found_field.endswith('_pg'):
            if '_2d' in field:
                return _wrap_2d_grouped(obj)
            if '_1d' in field:
                return _wrap_1d_grouped(obj, field.replace('_pg', '').replace('_1d', ''))
            if obj.ndim == 2:
                return _wrap_2d_grouped(obj)
            if obj.ndim == 1:
                return _wrap_1d_grouped(obj, field.replace('_pg', ''))
        if found_field.endswith('_pc'):
            if '_2d' in field:
                return _wrap_2d(obj)
            if '_1d' in field:
                return _wrap_1d(obj, field.replace('_pc', '').replace('_1d', ''))
            if obj.ndim == 2:
                return _wrap_2d(obj)
            if obj.ndim == 1:
                return _wrap_1d(obj, field.replace('_pc', ''))
        if not found_field.endswith('_records') and checks.is_np_array(obj):
            if is_grouped:
                if '_2d' in field:
                    return _wrap_2d_grouped(obj)
                if '_1d' in field:
                    return _wrap_1d_grouped(obj, field.replace('_1d', ''))
                if obj.shape == self.wrapper.get_shape():
                    return _wrap_2d_grouped(obj)
                if obj.shape == (self.wrapper.get_shape_2d()[1],):
                    return _wrap_1d_grouped(obj, field)
            else:
                if '_2d' in field:
                    return _wrap_2d(obj)
                if '_1d' in field:
                    return _wrap_1d(obj, field.replace('_1d', ''))
                if obj.shape == self.wrapper.shape:
                    return _wrap_2d(obj)
                if obj.shape == (self.wrapper.shape_2d[1],):
                    return _wrap_1d(obj, field)
        if force_wrapping:
            raise NotImplementedError(f"Cannot wrap field '{found_field}'")
        return obj

    def in_outputs_indexing_func(self,
                                 new_wrapper: ArrayWrapper,
                                 group_idxs: tp.MaybeArray,
                                 col_idxs: tp.Array1d) -> tp.Optional[tp.NamedTuple]:
        """Perform indexing on `Portfolio.in_outputs`.

        If the name of a field can be found as an attribute of `Portfolio`, reads this attribute's
        annotations to figure out the type and layout of the indexed object.

        !!! note
            Aliases are not supported.

        If not found, you can provide a grouping suffix to specify whether the array is
        grouped per column or group:

        * '_pcgs': per group if grouped with cash sharing, otherwise per column
        * '_pcg': per group if grouped, otherwise per column
        * '_pg': per group
        * '_pc': per column
        * '_records': records per column

        Additionally, you can provide a shape suffix to specify whether the object is
        a regular or reduced time series:

        * '_2d': element per timestamp and column/group (time series)
        * '_1d': element per column/group (reduced time series)

        Finally, looks for the field object having the same shape as that of `Portfolio.wrapper`,
        both grouped and ungrouped. Cash sharing plays no role in this case."""
        if self.in_outputs is None:
            return None

        in_outputs = self.in_outputs._asdict()
        new_in_outputs = {}
        cls = type(self)
        cls_dir = dir(cls)
        is_grouped = self.wrapper.grouper.is_grouped()

        def _index_1d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[group_idxs]

        def _index_1d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[col_idxs]

        def _index_2d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[:, group_idxs]

        def _index_2d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[:, col_idxs]

        def _index_records(obj: tp.RecordArray) -> tp.RecordArray:
            func = jit_reg.resolve_option(records_nb.col_map_nb, None)
            col_map = func(obj['col'], len(self.wrapper.columns))
            func = jit_reg.resolve_option(records_nb.record_col_map_select_nb, None)
            return func(obj, col_map, to_1d_array(col_idxs))

        for field, obj in in_outputs.items():
            if obj is None or isinstance(obj, bool):
                new_obj = obj
            else:
                new_obj = None
                if field in cls_dir:
                    method_or_prop = getattr(cls, field)
                    options = getattr(method_or_prop, 'options', {})
                    obj_type = options.get('obj_type', None)
                    group_by_aware = options.get('group_by_aware', None)
                    if obj_type is None:
                        raise TypeError(f"Cannot parse in-output '{field}': option 'obj_type' is missing")
                    if group_by_aware is None:
                        raise TypeError(f"Cannot parse in-output '{field}': option 'group_by_aware' is missing")

                    if obj_type == 'array':
                        if group_by_aware and is_grouped:
                            new_obj = _index_2d_by_group(obj)
                        else:
                            new_obj = _index_2d_by_col(obj)
                    elif obj_type == 'red_array':
                        if group_by_aware and is_grouped:
                            new_obj = _index_1d_by_group(obj)
                        else:
                            new_obj = _index_1d_by_col(obj)
                    elif obj_type == 'records':
                        new_obj = _index_records(obj)
                    else:
                        raise TypeError(f"Cannot index in-output '{field}': "
                                        f"option 'obj_type={obj_type}' not supported")
                else:
                    if '_pcgs' in field:
                        if '_2d' in field:
                            if is_grouped and self.cash_sharing:
                                new_obj = _index_2d_by_group(obj)
                            else:
                                new_obj = _index_2d_by_col(obj)
                        elif '_1d' in field:
                            if is_grouped and self.cash_sharing:
                                new_obj = _index_1d_by_group(obj)
                            else:
                                new_obj = _index_1d_by_col(obj)
                        elif obj.ndim == 2:
                            if is_grouped and self.cash_sharing:
                                new_obj = _index_2d_by_group(obj)
                            else:
                                new_obj = _index_2d_by_col(obj)
                        elif obj.ndim == 1:
                            if is_grouped and self.cash_sharing:
                                new_obj = _index_1d_by_group(obj)
                            else:
                                new_obj = _index_1d_by_col(obj)
                    elif field.endswith('_pcg'):
                        if '_2d' in field:
                            if is_grouped:
                                new_obj = _index_2d_by_group(obj)
                            else:
                                new_obj = _index_2d_by_col(obj)
                        elif '_1d' in field:
                            if is_grouped:
                                new_obj = _index_1d_by_group(obj)
                            else:
                                new_obj = _index_1d_by_col(obj)
                        elif obj.ndim == 2:
                            if is_grouped:
                                new_obj = _index_2d_by_group(obj)
                            else:
                                new_obj = _index_2d_by_col(obj)
                        elif obj.ndim == 1:
                            if is_grouped:
                                new_obj = _index_1d_by_group(obj)
                            else:
                                new_obj = _index_1d_by_col(obj)
                    elif field.endswith('_pg'):
                        if '_2d' in field:
                            new_obj = _index_2d_by_group(obj)
                        elif '_1d' in field:
                            new_obj = _index_1d_by_group(obj)
                        elif obj.ndim == 2:
                            new_obj = _index_2d_by_group(obj)
                        elif obj.ndim == 1:
                            new_obj = _index_1d_by_group(obj)
                    elif field.endswith('_pc'):
                        if '_2d' in field:
                            new_obj = _index_2d_by_col(obj)
                        elif '_1d' in field:
                            new_obj = _index_1d_by_col(obj)
                        elif obj.ndim == 2:
                            new_obj = _index_2d_by_col(obj)
                        elif obj.ndim == 1:
                            new_obj = _index_1d_by_col(obj)
                    elif field.endswith('_records'):
                        new_obj = _index_records(obj)
                    elif checks.is_np_array(obj):
                        if is_grouped:
                            if '_2d' in field:
                                new_obj = _index_2d_by_group(obj)
                            elif '_1d' in field:
                                new_obj = _index_1d_by_group(obj)
                            if obj.shape == self.wrapper.get_shape():
                                new_obj = _index_2d_by_group(obj)
                            elif obj.shape == (self.wrapper.get_shape_2d()[1],):
                                new_obj = _index_1d_by_group(obj)
                        else:
                            if '_2d' in field:
                                new_obj = _index_2d_by_col(obj)
                            elif '_1d' in field:
                                new_obj = _index_1d_by_col(obj)
                            if obj.shape == self.wrapper.shape:
                                new_obj = _index_2d_by_col(obj)
                            elif obj.shape == (self.wrapper.shape_2d[1],):
                                new_obj = _index_1d_by_col(obj)

                    if new_obj is None:
                        warnings.warn(f"Cannot figure out how to index in-output '{field}'. "
                                      f"Please provide a suffix.", stacklevel=2)

            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    # ############# Indexing ############# #

    def indexing_func(self: PortfolioT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> PortfolioT:
        """Perform indexing on `Portfolio`."""
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper.indexing_func_meta(pd_indexing_func, column_only_select=True, **kwargs)
        new_close = to_2d_array(self._close)[:, col_idxs]
        new_order_records = self.orders.get_by_col_idxs(col_idxs)
        new_log_records = self.logs.get_by_col_idxs(col_idxs)
        if isinstance(self._init_cash, int):
            new_init_cash = self._init_cash
        else:
            new_init_cash = to_1d_array(self._init_cash)
            if new_init_cash.shape[0] > 1:
                if self.cash_sharing:
                    new_init_cash = new_init_cash[group_idxs]
                else:
                    new_init_cash = new_init_cash[col_idxs]
        new_init_position = to_1d_array(self._init_position)
        if new_init_position.shape[0] > 1:
            new_init_position = new_init_position[col_idxs]
        new_cash_deposits = to_2d_array(self._cash_deposits)
        if new_cash_deposits.shape[1] > 1:
            if self.cash_sharing:
                new_cash_deposits = new_cash_deposits[:, group_idxs]
            else:
                new_cash_deposits = new_cash_deposits[:, col_idxs]
        new_cash_earnings = to_2d_array(self._cash_earnings)
        if new_cash_earnings.shape[1] > 1:
            new_cash_earnings = new_cash_earnings[:, col_idxs]
        if self._call_seq is not None:
            call_seq = to_2d_array(self._call_seq)
            new_call_seq = call_seq[:, col_idxs]
        else:
            new_call_seq = None
        if self._bm_close is not None and not isinstance(self._bm_close, bool):
            bm_close = to_2d_array(self._bm_close)
            new_bm_close = bm_close[:, col_idxs]
        else:
            new_bm_close = self._bm_close
        new_in_outputs = self.in_outputs_indexing_func(new_wrapper, group_idxs, col_idxs)

        return self.replace(
            wrapper=new_wrapper,
            close=new_close,
            order_records=new_order_records,
            log_records=new_log_records,
            init_cash=new_init_cash,
            init_position=new_init_position,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            call_seq=new_call_seq,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close
        )

    # ############# Class methods ############# #

    @classmethod
    def from_orders(cls: tp.Type[PortfolioT],
                    close: tp.ArrayLike,
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
                    open: tp.ArrayLike = np.nan,
                    high: tp.ArrayLike = np.nan,
                    low: tp.ArrayLike = np.nan,
                    init_cash: tp.Optional[tp.ArrayLike] = None,
                    init_position: tp.Optional[tp.ArrayLike] = None,
                    cash_deposits: tp.Optional[tp.ArrayLike] = None,
                    cash_earnings: tp.Optional[tp.ArrayLike] = None,
                    cash_dividends: tp.Optional[tp.ArrayLike] = None,
                    cash_sharing: tp.Optional[bool] = None,
                    call_seq: tp.Optional[tp.ArrayLike] = None,
                    attach_call_seq: tp.Optional[bool] = None,
                    ffill_val_price: tp.Optional[bool] = None,
                    update_value: tp.Optional[bool] = None,
                    max_orders: tp.Optional[int] = None,
                    max_logs: tp.Optional[int] = None,
                    seed: tp.Optional[int] = None,
                    group_by: tp.GroupByLike = None,
                    broadcast_kwargs: tp.KwargsLike = None,
                    jitted: tp.JittedOption = None,
                    chunked: tp.ChunkedOption = None,
                    wrapper_kwargs: tp.KwargsLike = None,
                    freq: tp.Optional[tp.FrequencyLike] = None,
                    bm_close: tp.Optional[tp.ArrayLike] = None,
                    **kwargs) -> PortfolioT:
        """Simulate portfolio from orders - size, price, fees, and other information.

        See `vectorbtpro.portfolio.nb.from_orders.simulate_from_orders_nb`.

        Args:
            close (array_like): Latest asset price at each time step.
                Will broadcast.

                Used for calculating unrealized PnL and portfolio value.
            size (float or array_like): Size to order.
                See `vectorbtpro.portfolio.enums.Order.size`. Will broadcast.
            size_type (SizeType or array_like): See `vectorbtpro.portfolio.enums.SizeType` and
                `vectorbtpro.portfolio.enums.Order.size_type`. Will broadcast.
            direction (Direction or array_like): See `vectorbtpro.portfolio.enums.Direction` and
                `vectorbtpro.portfolio.enums.Order.direction`. Will broadcast.
            price (array_like of float): Order price.
                See `vectorbtpro.portfolio.enums.Order.price`. Defaults to `np.inf`. Will broadcast.

                !!! note
                    Make sure to use the same timestamp for all order prices in the group with cash sharing
                    and `call_seq` set to `CallSeqType.Auto`.
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
                Defaults to `np.inf`. Will broadcast.

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
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                Each value in this sequence must indicate the position of column in the group to
                call next. Processing of `call_seq` goes always from left to right.
                For example, `[2, 0, 1]` would first call column 'c', then 'a', and finally 'b'.

                * Use `vectorbtpro.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

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
            max_orders (int): The max number of order records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            max_logs (int): The max number of log records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape if any of the `log` is True,
                otherwise to 0.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            group_by (any): Group columns. See `vectorbtpro.base.grouping.Grouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (any): See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            freq (any): Index frequency in case it cannot be parsed from `close`.
            bm_close (array_like): Latest benchmark price at each time step.
                Will broadcast.

                If not provided, will use `close`. If False, will not use any benchmark.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`.

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

            >>> pf.get_asset_value(group_by=False).vbt.plot()
            ```

            ![](/assets/images/simulate_nb_example.svg)

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
        portfolio_cfg = settings['portfolio']

        if size is None:
            size = portfolio_cfg['size']
        if size_type is None:
            size_type = portfolio_cfg['size_type']
        if direction is None:
            direction = portfolio_cfg['direction']
        if price is None:
            price = portfolio_cfg['price']
        if size is None:
            size = portfolio_cfg['size']
        if fees is None:
            fees = portfolio_cfg['fees']
        if fixed_fees is None:
            fixed_fees = portfolio_cfg['fixed_fees']
        if slippage is None:
            slippage = portfolio_cfg['slippage']
        if min_size is None:
            min_size = portfolio_cfg['min_size']
        if max_size is None:
            max_size = portfolio_cfg['max_size']
        if size_granularity is None:
            size_granularity = portfolio_cfg['size_granularity']
        if reject_prob is None:
            reject_prob = portfolio_cfg['reject_prob']
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg['price_area_vio_mode']
        if lock_cash is None:
            lock_cash = portfolio_cfg['lock_cash']
        if allow_partial is None:
            allow_partial = portfolio_cfg['allow_partial']
        if raise_reject is None:
            raise_reject = portfolio_cfg['raise_reject']
        if log is None:
            log = portfolio_cfg['log']
        if val_price is None:
            val_price = portfolio_cfg['val_price']
        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg['init_position']
        if cash_deposits is None:
            cash_deposits = portfolio_cfg['cash_deposits']
        if cash_earnings is None:
            cash_earnings = portfolio_cfg['cash_earnings']
        if cash_dividends is None:
            cash_dividends = portfolio_cfg['cash_dividends']
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']
        if cash_sharing and group_by is None:
            group_by = True
        if call_seq is None:
            call_seq = portfolio_cfg['call_seq']
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']
        if update_value is None:
            update_value = portfolio_cfg['update_value']
        if seed is None:
            seed = portfolio_cfg['seed']
        if seed is not None:
            set_seed(seed)
        if group_by is None:
            group_by = portfolio_cfg['group_by']
        if freq is None:
            freq = portfolio_cfg['freq']
        broadcast_kwargs = merge_dicts(portfolio_cfg['broadcast_kwargs'], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get('require_kwargs', {})
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        if bm_close is None:
            bm_close = portfolio_cfg['bm_close']

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
        )
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args['bm_close'] = bm_close
        broadcast_kwargs = merge_dicts(dict(
            keep_flex=dict(
                close=False,
                bm_close=False,
                _default=True
            )
        ), broadcast_kwargs)
        broadcasted_args = broadcast(broadcastable_args, **broadcast_kwargs)
        cash_earnings = broadcasted_args.pop('cash_earnings')
        cash_dividends = broadcasted_args.pop('cash_dividends')
        close = broadcasted_args['close']
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = broadcasted_args['bm_close']
            if not checks.is_pandas(bm_close):
                bm_close = pd.Series(bm_close) if bm_close.ndim == 1 else pd.DataFrame(bm_close)
            broadcasted_args['bm_close'] = to_2d_array(bm_close)
        else:
            broadcasted_args['bm_close'] = None
        flex_2d = close.ndim == 2
        broadcasted_args['close'] = to_2d_array(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)

        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            to_2d_array(cash_deposits, expand_axis=int(not flex_2d)),
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            to_pd=False,
            keep_flex=True,
            **require_kwargs
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_any_array(call_seq):
            call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if not np.any(log):
            max_logs = 0

        # Convert strings to numbers
        broadcasted_args['size_type'] = map_enum_fields(
            broadcasted_args['size_type'], SizeType)
        broadcasted_args['direction'] = map_enum_fields(
            broadcasted_args['direction'], Direction)
        broadcasted_args['price_area_vio_mode'] = map_enum_fields(
            broadcasted_args['price_area_vio_mode'], PriceAreaVioMode)

        # Check data types
        checks.assert_subdtype(cs_group_lens, np.int_)
        checks.assert_subdtype(call_seq, np.int_)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(cash_dividends, np.number)
        checks.assert_subdtype(broadcasted_args['size'], np.number)
        checks.assert_subdtype(broadcasted_args['price'], np.number)
        checks.assert_subdtype(broadcasted_args['size_type'], np.int_)
        checks.assert_subdtype(broadcasted_args['direction'], np.int_)
        checks.assert_subdtype(broadcasted_args['fees'], np.number)
        checks.assert_subdtype(broadcasted_args['fixed_fees'], np.number)
        checks.assert_subdtype(broadcasted_args['slippage'], np.number)
        checks.assert_subdtype(broadcasted_args['min_size'], np.number)
        checks.assert_subdtype(broadcasted_args['max_size'], np.number)
        checks.assert_subdtype(broadcasted_args['size_granularity'], np.number)
        checks.assert_subdtype(broadcasted_args['reject_prob'], np.number)
        checks.assert_subdtype(broadcasted_args['price_area_vio_mode'], np.int_)
        checks.assert_subdtype(broadcasted_args['lock_cash'], np.bool_)
        checks.assert_subdtype(broadcasted_args['allow_partial'], np.bool_)
        checks.assert_subdtype(broadcasted_args['raise_reject'], np.bool_)
        checks.assert_subdtype(broadcasted_args['log'], np.bool_)
        checks.assert_subdtype(broadcasted_args['val_price'], np.number)
        checks.assert_subdtype(broadcasted_args['open'], np.number)
        checks.assert_subdtype(broadcasted_args['high'], np.number)
        checks.assert_subdtype(broadcasted_args['low'], np.number)
        checks.assert_subdtype(broadcasted_args['close'], np.number)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args['bm_close'], np.number)

        # Remove arguments
        broadcasted_args.pop('bm_close', None)

        # Perform the simulation
        func = jit_reg.resolve_option(nb.simulate_from_orders_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        sim_out = func(
            target_shape=target_shape_2d,
            group_lens=cs_group_lens,  # group only if cash sharing is enabled to speed up
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            cash_dividends=cash_dividends,
            **broadcasted_args,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            flex_2d=flex_2d
        )

        # Create an instance
        return cls(
            wrapper,
            close,
            sim_out.order_records,
            sim_out.log_records,
            cash_sharing,
            init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs
        )

    @classmethod
    def from_signals(cls: tp.Type[PortfolioT],
                     close: tp.ArrayLike,
                     entries: tp.Optional[tp.ArrayLike] = None,
                     exits: tp.Optional[tp.ArrayLike] = None,
                     short_entries: tp.Optional[tp.ArrayLike] = None,
                     short_exits: tp.Optional[tp.ArrayLike] = None,
                     signal_func_nb: nb.SignalFuncT = nb.no_signal_func_nb,
                     signal_args: tp.ArgsLike = (),
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
                     accumulate: tp.Optional[tp.ArrayLike] = None,
                     upon_long_conflict: tp.Optional[tp.ArrayLike] = None,
                     upon_short_conflict: tp.Optional[tp.ArrayLike] = None,
                     upon_dir_conflict: tp.Optional[tp.ArrayLike] = None,
                     upon_opposite_entry: tp.Optional[tp.ArrayLike] = None,
                     direction: tp.Optional[tp.ArrayLike] = None,
                     val_price: tp.Optional[tp.ArrayLike] = None,
                     open: tp.ArrayLike = np.nan,
                     high: tp.ArrayLike = np.nan,
                     low: tp.ArrayLike = np.nan,
                     sl_stop: tp.Optional[tp.ArrayLike] = None,
                     sl_trail: tp.Optional[tp.ArrayLike] = None,
                     tp_stop: tp.Optional[tp.ArrayLike] = None,
                     stop_entry_price: tp.Optional[tp.ArrayLike] = None,
                     stop_exit_price: tp.Optional[tp.ArrayLike] = None,
                     upon_stop_exit: tp.Optional[tp.ArrayLike] = None,
                     upon_stop_update: tp.Optional[tp.ArrayLike] = None,
                     signal_priority: tp.Optional[tp.ArrayLike] = None,
                     adjust_sl_func_nb: nb.AdjustSLFuncT = nb.no_adjust_sl_func_nb,
                     adjust_sl_args: tp.Args = (),
                     adjust_tp_func_nb: nb.AdjustTPFuncT = nb.no_adjust_tp_func_nb,
                     adjust_tp_args: tp.Args = (),
                     use_stops: tp.Optional[bool] = None,
                     init_cash: tp.Optional[tp.ArrayLike] = None,
                     init_position: tp.Optional[tp.ArrayLike] = None,
                     cash_deposits: tp.Optional[tp.ArrayLike] = None,
                     cash_earnings: tp.Optional[tp.ArrayLike] = None,
                     cash_dividends: tp.Optional[tp.ArrayLike] = None,
                     cash_sharing: tp.Optional[bool] = None,
                     call_seq: tp.Optional[tp.ArrayLike] = None,
                     attach_call_seq: tp.Optional[bool] = None,
                     ffill_val_price: tp.Optional[bool] = None,
                     update_value: tp.Optional[bool] = None,
                     max_orders: tp.Optional[int] = None,
                     max_logs: tp.Optional[int] = None,
                     seed: tp.Optional[int] = None,
                     group_by: tp.GroupByLike = None,
                     broadcast_named_args: tp.KwargsLike = None,
                     broadcast_kwargs: tp.KwargsLike = None,
                     template_mapping: tp.Optional[tp.Mapping] = None,
                     jitted: tp.JittedOption = None,
                     chunked: tp.ChunkedOption = None,
                     wrapper_kwargs: tp.KwargsLike = None,
                     freq: tp.Optional[tp.FrequencyLike] = None,
                     bm_close: tp.Optional[tp.ArrayLike] = None,
                     **kwargs) -> PortfolioT:
        """Simulate portfolio from entry and exit signals.

        See `vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb`.

        You have three options to provide signals:

        1. `entries` and `exits`: The direction of each pair of signals is taken from `direction` argument.
            Best to use when the direction doesn't change throughout time.

            Uses `vectorbtpro.portfolio.nb.from_signals.dir_enex_signal_func_nb` as `signal_func_nb`.

            !!! hint
                `entries` and `exits` can be easily translated to direction-aware signals:

                * (True, True, 'longonly') -> True, True, False, False
                * (True, True, 'shortonly') -> False, False, True, True
                * (True, True, 'both') -> True, False, True, False

        2. `entries` (acting as long), `exits` (acting as long), `short_entries`, and `short_exits`:
            The direction is already built into the arrays. Best to use when the direction changes frequently
            (for example, if you have one indicator providing long signals and one providing short signals).

            Uses `vectorbtpro.portfolio.nb.from_signals.ls_enex_signal_func_nb` as `signal_func_nb`.

        3. `signal_func_nb` and `signal_args`: Custom signal function that returns direction-aware signals.
            Best to use when signals should be placed dynamically based on custom conditions.

        Args:
            close (array_like): See `Portfolio.from_orders`.
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
            signal_func_nb (callable): Function called to generate signals.

                Must accept `vectorbtpro.portfolio.enums.SignalContext` and `*signal_args`.
                Must return long entry signal, long exit signal, short entry signal, and short exit signal.

                !!! note
                    Stop signal has priority: `signal_func_nb` is executed only if there is no stop signal.
            signal_args (tuple): Packed arguments passed to `signal_func_nb`.
                Defaults to `()`.
            size (float or array_like): See `Portfolio.from_orders`.

                !!! note
                    Negative size is not allowed. You must express direction using signals.
            size_type (SizeType or array_like): See `Portfolio.from_orders`.

                Only `SizeType.Amount`, `SizeType.Value`, and `SizeType.Percent` are supported.
                Other modes such as target percentage are not compatible with signals since
                their logic may contradict the direction of the signal.

                !!! note
                    `SizeType.Percent` does not support position reversal. Switch to a single
                    direction or use `vectorbtpro.portfolio.enums.OppositeEntryMode.Close` to close the position first.

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
            accumulate (bool, AccumulationMode or array_like): See `vectorbtpro.portfolio.enums.AccumulationMode`.
                If True, becomes 'both'. If False, becomes 'disabled'. Will broadcast.

                When enabled, `Portfolio.from_signals` behaves similarly to `Portfolio.from_orders`.
            upon_long_conflict (ConflictMode or array_like): Conflict mode for long signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_short_conflict (ConflictMode or array_like): Conflict mode for short signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_dir_conflict (DirectionConflictMode or array_like): See `vectorbtpro.portfolio.enums.DirectionConflictMode`. Will broadcast.
            upon_opposite_entry (OppositeEntryMode or array_like): See `vectorbtpro.portfolio.enums.OppositeEntryMode`. Will broadcast.
            direction (Direction or array_like): See `Portfolio.from_orders`.

                Takes only effect if `short_entries` and `short_exits` are not set.
            val_price (array_like of float): See `Portfolio.from_orders`.
            open (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` gets replaced by `close`.
            high (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the maximum out of `open` and `close`.
            low (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the minimum out of `open` and `close`.
            sl_stop (array_like of float): Stop loss.
                Will broadcast.

                A percentage below/above the acquisition price for long/short position.
                Note that 0.01 = 1%.
            sl_trail (array_like of bool): Whether `sl_stop` should be trailing.
                Will broadcast.
            tp_stop (array_like of float): Take profit.
                Will broadcast.

                A percentage above/below the acquisition price for long/short position.
                Note that 0.01 = 1%.
            stop_entry_price (StopEntryPrice or array_like): See `vectorbtpro.portfolio.enums.StopEntryPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry.
            stop_exit_price (StopExitPrice or array_like): See `vectorbtpro.portfolio.enums.StopExitPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon exit.
            upon_stop_exit (StopExitMode or array_like): See `vectorbtpro.portfolio.enums.StopExitMode`.
                Will broadcast.

                If provided on per-element basis, gets applied upon exit.
            upon_stop_update (StopUpdateMode or array_like): See `vectorbtpro.portfolio.enums.StopUpdateMode`.
                Will broadcast.

                Only has effect if accumulation is enabled.

                If provided on per-element basis, gets applied upon repeated entry.
            signal_priority (SignalPriority or array_like): See `vectorbtpro.portfolio.enums.SignalPriority`.
                Will broadcast.

                Only has effect if both stop signal and user-defined signal are executable.

                !!! note
                    Which option to choose depends on **when** the user-defined signal is available:

                    * at open (user signal wins)
                    * between open and close (not enough information!)
                    * at close (stop signal wins)
            adjust_sl_func_nb (callable): Function to adjust stop loss.
                Defaults to `vectorbtpro.portfolio.nb.from_signals.no_adjust_sl_func_nb`.

                Called for each element before each row.

                Must accept `vectorbtpro.portfolio.enums.AdjustSLContext` and `*adjust_sl_args`.
                Must return a tuple of a new stop value and trailing flag.
            adjust_sl_args (tuple): Packed arguments passed to `adjust_sl_func_nb`.
                Defaults to `()`.
            adjust_tp_func_nb (callable): Function to adjust take profit.
                Defaults to `vectorbtpro.portfolio.nb.from_signals.no_adjust_tp_func_nb`.

                Called for each element before each row.

                Must accept `vectorbtpro.portfolio.enums.AdjustTPContext` and `*adjust_tp_args`.
                of the stop, and `*adjust_tp_args`. Must return a new stop value.
            adjust_tp_args (tuple): Packed arguments passed to `adjust_tp_func_nb`.
                Defaults to `()`.
            use_stops (bool): Whether to use stops.
                Defaults to None, which becomes True if any of the stops are not NaN or
                any of the adjustment functions are custom.

                Disable this to make simulation a bit faster for simple use cases.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_dividends (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): See `Portfolio.from_orders`.
            call_seq (CallSeqType or array_like): See `Portfolio.from_orders`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            ffill_val_price (bool): See `Portfolio.from_orders`.
            update_value (bool): See `Portfolio.from_orders`.
            max_orders (int): See `Portfolio.from_orders`.
            max_logs (int): See `Portfolio.from_orders`.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): Dictionary with named arguments to broadcast.

                You can then pass argument names wrapped with `vectorbtpro.utils.template.Rep`
                and this method will substitute them by their corresponding broadcasted objects.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_mapping (mapping): Mapping to replace templates in arguments.
            jitted (any): See `Portfolio.from_orders`.
            chunked (any): See `Portfolio.from_orders`.
            wrapper_kwargs (dict): See `Portfolio.from_orders`.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`.

        !!! note
            Stop signal has priority - it's executed before other signals within the same bar.
            That is, if a stop signal is present, no other signals are generated and executed
            since there is a limit of one order per symbol and bar.

        !!! hint
            If you generated signals using close price, don't forget to shift your signals by one tick
            forward, for example, with `signals.vbt.fshift(1)`. In general, make sure to use a price
            that comes after the signal.

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
            ...     broadcast_kwargs=dict(columns_from=pd.Index(Direction._fields, name='direction')))
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
            ...     sl_stop=0.1, sl_trail=True, tp_stop=0.2)  # take profit hit
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
            ...     sl_stop=0.1, sl_trail=True, tp_stop=0.3)  # stop loss hit
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
            ...     sl_stop=np.inf, sl_trail=True, tp_stop=np.inf)  # nothing hit, exit as usual
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4     0.0
            5   -10.0
            dtype: float64
            ```

            * Test all stop combinations:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     sl_stop=pd.Index([0.1, 0.2]),
            ...     sl_trail=pd.Index([False, True]),
            ...     tp_stop=pd.Index([0.2, 0.3])
            ... )
            >>> pf.asset_flow
            sl_stop                      0.1                     0.2
            sl_trail       False        True       False        True
            tp_stop    0.2   0.3   0.2   0.3   0.2   0.3   0.2   0.3
            0         10.0  10.0  10.0  10.0  10.0  10.0  10.0  10.0
            1          0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            2        -10.0   0.0 -10.0   0.0 -10.0   0.0 -10.0   0.0
            3          0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            4          0.0   0.0   0.0 -10.0   0.0   0.0   0.0   0.0
            5          0.0 -10.0   0.0   0.0   0.0 -10.0   0.0 -10.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            * We can implement our own stop loss or take profit, or adjust the existing one at each time step.
            Let's implement [stepped stop-loss](https://www.freqtrade.io/en/stable/strategy-advanced/#stepped-stoploss):

            ```pycon
            >>> @njit
            ... def adjust_sl_func_nb(c):
            ...     current_profit = (c.val_price_now - c.init_price) / c.init_price
            ...     if current_profit >= 0.40:
            ...         return 0.25, True
            ...     elif current_profit >= 0.25:
            ...         return 0.15, True
            ...     elif current_profit >= 0.20:
            ...         return 0.07, True
            ...     return c.curr_stop, c.curr_trail

            >>> close = pd.Series([10, 11, 12, 11, 10])
            >>> pf = vbt.Portfolio.from_signals(close, adjust_sl_func_nb=adjust_sl_func_nb)
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0  # 7% from 12 hit
            4    11.0
            dtype: float64
            ```

            * Sometimes there is a need to provide or transform signals dynamically. For this, we can implement
            a custom signal function `signal_func_nb`. For example, let's implement a signal function that
            takes two numerical arrays - long and short one - and transforms them into 4 direction-aware boolean
            arrays that vectorbt understands:

            ```pycon
            >>> @njit
            ... def signal_func_nb(c, long_num_arr, short_num_arr):
            ...     long_num = nb.get_elem_nb(c, long_num_arr)
            ...     short_num = nb.get_elem_nb(c, short_num_arr)
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
        portfolio_cfg = settings['portfolio']

        ls_mode = short_entries is not None or short_exits is not None
        signal_func_mode = signal_func_nb is not nb.no_signal_func_nb
        if (entries is not None or exits is not None or ls_mode) and signal_func_mode:
            raise ValueError("Either any of the signal arrays or signal_func_nb must be provided, not both")
        if entries is None:
            if exits is None and not ls_mode:
                entries = True
            else:
                entries = False
        if exits is None:
            exits = False
        if short_entries is None:
            short_entries = False
        if short_exits is None:
            short_exits = False
        if signal_func_nb is nb.no_signal_func_nb:
            if ls_mode:
                signal_func_nb = nb.ls_enex_signal_func_nb
            else:
                signal_func_nb = nb.dir_enex_signal_func_nb
        if size is None:
            size = portfolio_cfg['size']
        if size_type is None:
            size_type = portfolio_cfg['size_type']
        if price is None:
            price = portfolio_cfg['price']
        if fees is None:
            fees = portfolio_cfg['fees']
        if fixed_fees is None:
            fixed_fees = portfolio_cfg['fixed_fees']
        if slippage is None:
            slippage = portfolio_cfg['slippage']
        if min_size is None:
            min_size = portfolio_cfg['min_size']
        if max_size is None:
            max_size = portfolio_cfg['max_size']
        if size_granularity is None:
            size_granularity = portfolio_cfg['size_granularity']
        if reject_prob is None:
            reject_prob = portfolio_cfg['reject_prob']
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg['price_area_vio_mode']
        if lock_cash is None:
            lock_cash = portfolio_cfg['lock_cash']
        if allow_partial is None:
            allow_partial = portfolio_cfg['allow_partial']
        if raise_reject is None:
            raise_reject = portfolio_cfg['raise_reject']
        if log is None:
            log = portfolio_cfg['log']
        if accumulate is None:
            accumulate = portfolio_cfg['accumulate']
        if upon_long_conflict is None:
            upon_long_conflict = portfolio_cfg['upon_long_conflict']
        if upon_short_conflict is None:
            upon_short_conflict = portfolio_cfg['upon_short_conflict']
        if upon_dir_conflict is None:
            upon_dir_conflict = portfolio_cfg['upon_dir_conflict']
        if upon_opposite_entry is None:
            upon_opposite_entry = portfolio_cfg['upon_opposite_entry']
        if direction is not None and ls_mode:
            warnings.warn("direction has no effect if short_entries and short_exits are set", stacklevel=2)
        if direction is None:
            direction = portfolio_cfg['signal_direction']
        if val_price is None:
            val_price = portfolio_cfg['val_price']
        if sl_stop is None:
            sl_stop = portfolio_cfg['sl_stop']
        if sl_trail is None:
            sl_trail = portfolio_cfg['sl_trail']
        if tp_stop is None:
            tp_stop = portfolio_cfg['tp_stop']
        if stop_entry_price is None:
            stop_entry_price = portfolio_cfg['stop_entry_price']
        if stop_exit_price is None:
            stop_exit_price = portfolio_cfg['stop_exit_price']
        if upon_stop_exit is None:
            upon_stop_exit = portfolio_cfg['upon_stop_exit']
        if upon_stop_update is None:
            upon_stop_update = portfolio_cfg['upon_stop_update']
        if signal_priority is None:
            signal_priority = portfolio_cfg['signal_priority']
        if use_stops is None:
            use_stops = portfolio_cfg['use_stops']
        if use_stops is None:
            if isinstance(sl_stop, float) and \
                    np.isnan(sl_stop) and \
                    isinstance(tp_stop, float) and \
                    np.isnan(tp_stop) and \
                    adjust_sl_func_nb == nb.no_adjust_sl_func_nb and \
                    adjust_tp_func_nb == nb.no_adjust_tp_func_nb:
                use_stops = False
            else:
                use_stops = True

        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg['init_position']
        if cash_deposits is None:
            cash_deposits = portfolio_cfg['cash_deposits']
        if cash_earnings is None:
            cash_earnings = portfolio_cfg['cash_earnings']
        if cash_dividends is None:
            cash_dividends = portfolio_cfg['cash_dividends']
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']
        if cash_sharing and group_by is None:
            group_by = True
        if call_seq is None:
            call_seq = portfolio_cfg['call_seq']
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']
        if update_value is None:
            update_value = portfolio_cfg['update_value']
        if seed is None:
            seed = portfolio_cfg['seed']
        if seed is not None:
            set_seed(seed)
        if group_by is None:
            group_by = portfolio_cfg['group_by']
        if freq is None:
            freq = portfolio_cfg['freq']
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_kwargs = merge_dicts(portfolio_cfg['broadcast_kwargs'], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get('require_kwargs', {})
        template_mapping = merge_dicts(portfolio_cfg['template_mapping'], template_mapping)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        if bm_close is None:
            bm_close = portfolio_cfg['bm_close']

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
            accumulate=accumulate,
            upon_long_conflict=upon_long_conflict,
            upon_short_conflict=upon_short_conflict,
            upon_dir_conflict=upon_dir_conflict,
            upon_opposite_entry=upon_opposite_entry,
            val_price=val_price,
            open=open,
            high=high,
            low=low,
            close=close,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            tp_stop=tp_stop,
            stop_entry_price=stop_entry_price,
            stop_exit_price=stop_exit_price,
            upon_stop_exit=upon_stop_exit,
            upon_stop_update=upon_stop_update,
            signal_priority=signal_priority
        )
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args['bm_close'] = bm_close
        if not signal_func_mode:
            if ls_mode:
                broadcastable_args['entries'] = entries
                broadcastable_args['exits'] = exits
                broadcastable_args['short_entries'] = short_entries
                broadcastable_args['short_exits'] = short_exits
            else:
                broadcastable_args['entries'] = entries
                broadcastable_args['exits'] = exits
                broadcastable_args['direction'] = direction
        broadcastable_args = {**broadcastable_args, **broadcast_named_args}
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        broadcast_kwargs = merge_dicts(dict(
            keep_flex=dict(
                close=False,
                bm_close=False,
                _default=True
            )
        ), broadcast_kwargs)
        broadcasted_args = broadcast(broadcastable_args, **broadcast_kwargs)
        cash_earnings = broadcasted_args.pop('cash_earnings')
        cash_dividends = broadcasted_args.pop('cash_dividends')
        close = broadcasted_args['close']
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = broadcasted_args['bm_close']
            if not checks.is_pandas(bm_close):
                bm_close = pd.Series(bm_close) if bm_close.ndim == 1 else pd.DataFrame(bm_close)
            broadcasted_args['bm_close'] = to_2d_array(bm_close)
        else:
            broadcasted_args['bm_close'] = None
        flex_2d = close.ndim == 2
        broadcasted_args['close'] = to_2d_array(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)

        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            to_2d_array(cash_deposits, expand_axis=int(not flex_2d)),
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            to_pd=False,
            keep_flex=True,
            **require_kwargs
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_any_array(call_seq):
            call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if not np.any(log):
            max_logs = 0

        # Convert strings to numbers
        broadcasted_args['size_type'] = map_enum_fields(
            broadcasted_args['size_type'], SizeType)
        broadcasted_args['price_area_vio_mode'] = map_enum_fields(
            broadcasted_args['price_area_vio_mode'], PriceAreaVioMode)
        broadcasted_args['accumulate'] = map_enum_fields(
            broadcasted_args['accumulate'], AccumulationMode, ignore_type=(int, bool))
        broadcasted_args['upon_long_conflict'] = map_enum_fields(
            broadcasted_args['upon_long_conflict'], ConflictMode)
        broadcasted_args['upon_short_conflict'] = map_enum_fields(
            broadcasted_args['upon_short_conflict'], ConflictMode)
        broadcasted_args['upon_dir_conflict'] = map_enum_fields(
            broadcasted_args['upon_dir_conflict'], DirectionConflictMode)
        broadcasted_args['upon_opposite_entry'] = map_enum_fields(
            broadcasted_args['upon_opposite_entry'], OppositeEntryMode)
        broadcasted_args['stop_entry_price'] = map_enum_fields(
            broadcasted_args['stop_entry_price'], StopEntryPrice)
        broadcasted_args['stop_exit_price'] = map_enum_fields(
            broadcasted_args['stop_exit_price'], StopExitPrice)
        broadcasted_args['upon_stop_exit'] = map_enum_fields(
            broadcasted_args['upon_stop_exit'], StopExitMode)
        broadcasted_args['upon_stop_update'] = map_enum_fields(
            broadcasted_args['upon_stop_update'], StopUpdateMode)
        broadcasted_args['signal_priority'] = map_enum_fields(
            broadcasted_args['signal_priority'], SignalPriority)
        if 'direction' in broadcasted_args:
            broadcasted_args['direction'] = map_enum_fields(
                broadcasted_args['direction'], Direction)

        # Check data types
        checks.assert_subdtype(cs_group_lens, np.int_)
        checks.assert_subdtype(call_seq, np.int_)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(cash_dividends, np.number)
        checks.assert_subdtype(broadcasted_args['size'], np.number)
        checks.assert_subdtype(broadcasted_args['price'], np.number)
        checks.assert_subdtype(broadcasted_args['size_type'], np.int_)
        checks.assert_subdtype(broadcasted_args['fees'], np.number)
        checks.assert_subdtype(broadcasted_args['fixed_fees'], np.number)
        checks.assert_subdtype(broadcasted_args['slippage'], np.number)
        checks.assert_subdtype(broadcasted_args['min_size'], np.number)
        checks.assert_subdtype(broadcasted_args['max_size'], np.number)
        checks.assert_subdtype(broadcasted_args['size_granularity'], np.number)
        checks.assert_subdtype(broadcasted_args['reject_prob'], np.number)
        checks.assert_subdtype(broadcasted_args['price_area_vio_mode'], np.int_)
        checks.assert_subdtype(broadcasted_args['lock_cash'], np.bool_)
        checks.assert_subdtype(broadcasted_args['allow_partial'], np.bool_)
        checks.assert_subdtype(broadcasted_args['raise_reject'], np.bool_)
        checks.assert_subdtype(broadcasted_args['log'], np.bool_)
        checks.assert_subdtype(broadcasted_args['accumulate'], (np.int_, np.bool_))
        checks.assert_subdtype(broadcasted_args['upon_long_conflict'], np.int_)
        checks.assert_subdtype(broadcasted_args['upon_short_conflict'], np.int_)
        checks.assert_subdtype(broadcasted_args['upon_dir_conflict'], np.int_)
        checks.assert_subdtype(broadcasted_args['upon_opposite_entry'], np.int_)
        checks.assert_subdtype(broadcasted_args['val_price'], np.number)
        checks.assert_subdtype(broadcasted_args['open'], np.number)
        checks.assert_subdtype(broadcasted_args['high'], np.number)
        checks.assert_subdtype(broadcasted_args['low'], np.number)
        checks.assert_subdtype(broadcasted_args['close'], np.number)
        checks.assert_subdtype(broadcasted_args['sl_stop'], np.number)
        checks.assert_subdtype(broadcasted_args['sl_trail'], np.bool_)
        checks.assert_subdtype(broadcasted_args['tp_stop'], np.number)
        checks.assert_subdtype(broadcasted_args['stop_entry_price'], np.int_)
        checks.assert_subdtype(broadcasted_args['stop_exit_price'], np.int_)
        checks.assert_subdtype(broadcasted_args['upon_stop_exit'], np.int_)
        checks.assert_subdtype(broadcasted_args['upon_stop_update'], np.int_)
        checks.assert_subdtype(broadcasted_args['signal_priority'], np.int_)
        if 'entries' in broadcasted_args:
            checks.assert_subdtype(broadcasted_args['entries'], np.bool_)
        if 'exits' in broadcasted_args:
            checks.assert_subdtype(broadcasted_args['exits'], np.bool_)
        if 'short_entries' in broadcasted_args:
            checks.assert_subdtype(broadcasted_args['short_entries'], np.bool_)
        if 'short_exits' in broadcasted_args:
            checks.assert_subdtype(broadcasted_args['short_exits'], np.bool_)
        if 'direction' in broadcasted_args:
            checks.assert_subdtype(broadcasted_args['direction'], np.int_)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args['bm_close'], np.number)

        # Prepare arguments
        template_mapping = merge_dicts(
            broadcasted_args,
            dict(
                target_shape=target_shape_2d,
                group_lens=cs_group_lens,
                call_seq=call_seq,
                init_cash=init_cash,
                init_position=init_position,
                cash_deposits=cash_deposits,
                cash_earnings=cash_earnings,
                cash_dividends=cash_dividends,
                adjust_sl_func_nb=adjust_sl_func_nb,
                adjust_sl_args=adjust_sl_args,
                adjust_tp_func_nb=adjust_tp_func_nb,
                adjust_tp_args=adjust_tp_args,
                use_stops=use_stops,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                max_orders=max_orders,
                max_logs=max_logs,
                flex_2d=flex_2d,
                wrapper=wrapper
            ),
            template_mapping
        )
        adjust_sl_args = deep_substitute(adjust_sl_args, template_mapping, sub_id='adjust_sl_args')
        adjust_tp_args = deep_substitute(adjust_tp_args, template_mapping, sub_id='adjust_tp_args')
        if signal_func_mode:
            signal_args = deep_substitute(signal_args, template_mapping, sub_id='signal_args')
        else:
            if ls_mode:
                signal_args = (
                    broadcasted_args.pop('entries'),
                    broadcasted_args.pop('exits'),
                    broadcasted_args.pop('short_entries'),
                    broadcasted_args.pop('short_exits')
                )
                chunked = ch.specialize_chunked_option(
                    chunked,
                    arg_take_spec=dict(
                        signal_args=ch.ArgsTaker(
                            portfolio_ch.flex_array_gl_slicer,
                            portfolio_ch.flex_array_gl_slicer,
                            portfolio_ch.flex_array_gl_slicer,
                            portfolio_ch.flex_array_gl_slicer
                        )
                    )
                )
            else:
                signal_args = (
                    broadcasted_args.pop('entries'),
                    broadcasted_args.pop('exits'),
                    broadcasted_args.pop('direction')
                )
                chunked = ch.specialize_chunked_option(
                    chunked,
                    arg_take_spec=dict(
                        signal_args=ch.ArgsTaker(
                            portfolio_ch.flex_array_gl_slicer,
                            portfolio_ch.flex_array_gl_slicer,
                            portfolio_ch.flex_array_gl_slicer
                        )
                    )
                )
        for k in broadcast_named_args:
            broadcasted_args.pop(k)
        broadcasted_args.pop('bm_close', None)

        # Perform the simulation
        func = jit_reg.resolve_option(nb.simulate_from_signal_func_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        sim_out = func(
            target_shape=target_shape_2d,
            group_lens=cs_group_lens,  # group only if cash sharing is enabled to speed up
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            cash_dividends=cash_dividends,
            signal_func_nb=signal_func_nb,
            signal_args=signal_args,
            **broadcasted_args,
            adjust_sl_func_nb=adjust_sl_func_nb,
            adjust_sl_args=adjust_sl_args,
            adjust_tp_func_nb=adjust_tp_func_nb,
            adjust_tp_args=adjust_tp_args,
            use_stops=use_stops,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            flex_2d=flex_2d
        )

        # Create an instance
        return cls(
            wrapper,
            close,
            sim_out.order_records,
            sim_out.log_records,
            cash_sharing,
            init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs
        )

    @classmethod
    def from_holding(cls: tp.Type[PortfolioT],
                     close: tp.ArrayLike,
                     size: tp.Optional[tp.ArrayLike] = None,
                     direction: tp.Optional[tp.ArrayLike] = None,
                     sell_at_end: tp.Optional[bool] = None,
                     base_method: tp.Optional[str] = None,
                     **kwargs) -> PortfolioT:
        """Simulate portfolio from plain holding.

        Has two base methods:

        * 'from_signals': Based on `Portfolio.from_signals`. Faster than the second method and allows
            using the native broadcasting mechanism of the underlying class method, but has less
            sizers available (e.g., there is no support for `vectorbtpro.portfolio.enums.SizeType.TargetPercent`).
        * 'from_orders': Based on `Portfolio.from_orders`. Allows using all implemented sizers,
            but requires conversion of `close` to pandas prior to broadcasting and must broadcast `size`
            to `close` to set all elements after the first timestamp to `np.nan`.

        `**kwargs` are passed to the underlying class method.

        For the default base method, see `hold_base_method` in `vectorbtpro._settings.portfolio`.

        !!! note
            If `sell_at_end` is True, will place an opposite signal at the very end.
            Be careful when using both directions, since this will most likely reverse the position.

        Usage:
            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_holding(close, base_method='from_signals')
            >>> pf.final_value
            500.0

            >>> pf = vbt.Portfolio.from_holding(close, base_method='from_orders')
            >>> pf.final_value
            500.0
            ```
        """
        from vectorbtpro._settings import settings
        portfolio_cfg = settings['portfolio']

        if direction is None:
            direction = portfolio_cfg['hold_direction']
        if sell_at_end is None:
            sell_at_end = portfolio_cfg['sell_at_end']
        if base_method is None:
            base_method = portfolio_cfg['hold_base_method']

        if base_method.lower() == 'from_signals':
            if sell_at_end:
                entries = broadcast_to(True, close, require_kwargs=dict(requirements='W'))
                exits = broadcast_to(False, close, require_kwargs=dict(requirements='W'))
                if checks.is_pandas(entries):
                    entries.iloc[1:] = False
                else:
                    entries[1:] = False
                if checks.is_pandas(exits):
                    exits.iloc[-1] = True
                else:
                    exits[-1] = True
            else:
                entries = True
                exits = False
            return cls.from_signals(
                close,
                entries=entries,
                exits=exits,
                accumulate=False,
                size=size,
                direction=direction,
                **kwargs
            )
        if base_method.lower() == 'from_orders':
            if size is None:
                size = portfolio_cfg['size']
            close = to_pd_array(close)
            size_arr = broadcast_to(size, close, require_kwargs=dict(requirements='W'))
            if checks.is_pandas(size_arr):
                size_arr.iloc[1:] = np.nan
                if sell_at_end:
                    size_arr.iloc[-1] = -size
            else:
                size_arr[1:] = np.nan
                if sell_at_end:
                    size_arr[-1] = -size
            return cls.from_orders(close, size=size_arr, direction=direction, **kwargs)
        raise ValueError(f"Unknown base method '{base_method}'")

    @classmethod
    def from_random_signals(cls: tp.Type[PortfolioT],
                            close: tp.ArrayLike,
                            n: tp.Optional[tp.ArrayLike] = None,
                            prob: tp.Optional[tp.ArrayLike] = None,
                            entry_prob: tp.Optional[tp.ArrayLike] = None,
                            exit_prob: tp.Optional[tp.ArrayLike] = None,
                            param_product: bool = False,
                            seed: tp.Optional[int] = None,
                            run_kwargs: tp.KwargsLike = None,
                            **kwargs) -> PortfolioT:
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
        portfolio_cfg = settings['portfolio']

        close = to_pd_array(close)
        close_wrapper = ArrayWrapper.from_obj(close)
        if entry_prob is None:
            entry_prob = prob
        if exit_prob is None:
            exit_prob = prob
        if seed is None:
            seed = portfolio_cfg['seed']
        if run_kwargs is None:
            run_kwargs = {}

        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or entry_prob and exit_prob must be provided")
        if n is not None:
            rand = RANDNX.run(
                n=n,
                input_shape=close.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs
            )
            entries = rand.entries
            exits = rand.exits
        elif entry_prob is not None and exit_prob is not None:
            rprobnx = RPROBNX.run(
                entry_prob=entry_prob,
                exit_prob=exit_prob,
                param_product=param_product,
                input_shape=close.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs
            )
            entries = rprobnx.entries
            exits = rprobnx.exits
        else:
            raise ValueError("At least n or entry_prob and exit_prob must be provided")

        return cls.from_signals(close, entries, exits, seed=seed, **kwargs)

    @classmethod
    def from_order_func(cls: tp.Type[PortfolioT],
                        close: tp.ArrayLike,
                        order_func_nb: tp.Union[nb.OrderFuncT, nb.FlexOrderFuncT],
                        *order_args,
                        flexible: tp.Optional[bool] = None,
                        init_cash: tp.Optional[tp.ArrayLike] = None,
                        init_position: tp.Optional[tp.ArrayLike] = None,
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
                        open: tp.ArrayLike = np.nan,
                        high: tp.ArrayLike = np.nan,
                        low: tp.ArrayLike = np.nan,
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
                        template_mapping: tp.Optional[tp.Mapping] = None,
                        keep_inout_raw: tp.Optional[bool] = None,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        wrapper_kwargs: tp.KwargsLike = None,
                        freq: tp.Optional[tp.FrequencyLike] = None,
                        bm_close: tp.Optional[tp.ArrayLike] = None,
                        **kwargs) -> PortfolioT:
        """Build portfolio from a custom order function.

        !!! hint
            See `vectorbtpro.portfolio.nb.from_order_func.simulate_nb` for illustrations and argument definitions.

        For more details on individual simulation functions:

        * not `row_wise` and not `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.simulate_nb`
        * not `row_wise` and `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.flex_simulate_nb`
        * `row_wise` and not `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.simulate_row_wise_nb`
        * `row_wise` and `flexible`: See `vectorbtpro.portfolio.nb.from_order_func.flex_simulate_row_wise_nb`

        Args:
            close (array_like): Latest asset price at each time step.
                Will broadcast.

                Used for calculating unrealized PnL and portfolio value.
            order_func_nb (callable): Order generation function.
            *order_args: Arguments passed to `order_func_nb`.
            flexible (bool): Whether to simulate using a flexible order function.

                This lifts the limit of one order per tick and symbol.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): Whether to share cash within the same group.

                If `group_by` is None, `group_by` becomes True to form a single group with cash sharing.
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                * Use `vectorbtpro.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

                !!! note
                    CallSeqType.Auto must be implemented manually.
                    Use `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_nb`
                    or `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_out_nb` in `pre_segment_func_nb`.
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
            template_mapping (mapping): See `Portfolio.from_signals`.
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
            wrapper_kwargs (dict): See `Portfolio.from_orders`.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        For defaults, see `vectorbtpro._settings.portfolio`.

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
            ...     return nb.order_nb(size=size)

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
            ...         return nb.close_position_nb()
            ...
            ...     if last_pos_state[0] == 1:
            ...         size = -np.inf  # open short
            ...         last_pos_state[0] = -1
            ...     else:
            ...         size = np.inf  # open long
            ...         last_pos_state[0] = 1
            ...     return nb.order_nb(size=size)

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
            ...         c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
            ...     nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            ...     return ()

            >>> @njit
            ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
            ...     return nb.order_nb(
            ...         size=nb.get_elem_nb(c, size),
            ...         price=nb.get_elem_nb(c, price),
            ...         size_type=nb.get_elem_nb(c, size_type),
            ...         direction=nb.get_elem_nb(c, direction),
            ...         fees=nb.get_elem_nb(c, fees),
            ...         fixed_fees=nb.get_elem_nb(c, fixed_fees),
            ...         slippage=nb.get_elem_nb(c, slippage)
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> size_template = vbt.RepEval('np.asarray(1 / group_lens[0])')

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
            ...         size_type=SizeType.TargetPercent,
            ...         direction=Direction.LongOnly,
            ...         fees=0.001,
            ...         fixed_fees=1.,
            ...         slippage=0.001
            ...     ),
            ...     template_mapping=dict(np=np),  # required by size_template
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot()
            ```

            ![](/assets/images/simulate_nb_example.svg)

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
            >>> pf['g2'].get_asset_value(group_by=False).vbt.plot()
            ```

            ![](/assets/images/from_order_func_g1.svg)

            ![](/assets/images/from_order_func_g2.svg)

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
            ...     entry_now = nb.get_elem_nb(c, entries)
            ...     exit_now = nb.get_elem_nb(c, exits)
            ...     size_now = nb.get_elem_nb(c, size)
            ...     price_now = nb.get_elem_nb(c, c.close)
            ...     stop_price_now = stop_price[c.col]
            ...
            ...     # Our logic
            ...     if entry_now:
            ...         if c.position_now == 0:
            ...             return nb.order_nb(
            ...                 size=size_now,
            ...                 price=price_now,
            ...                 direction=Direction.LongOnly)
            ...     elif exit_now or price_now >= stop_price_now:
            ...         if c.position_now > 0:
            ...             return nb.order_nb(
            ...                 size=-size_now,
            ...                 price=price_now,
            ...                 direction=Direction.LongOnly)
            ...     return NoOrder

            >>> @njit
            ... def post_order_func_nb(c, stop_price, stop):
            ...     # Same broadcasting as for size
            ...     stop_now = nb.get_elem_nb(c, stop)
            ...
            ...     if c.order_result.status == OrderStatus.Filled:
            ...         if c.order_result.side == OrderSide.Buy:
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
            ...         return c.from_col, nb.order_nb(size=size, price=c.open[c.i, c.from_col])
            ...     if c.call_idx == 1:
            ...         return c.from_col, nb.close_position_nb(price=c.close[c.i, c.from_col])
            ...     return -1, NoOrder

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
        portfolio_cfg = settings['portfolio']

        if flexible is None:
            flexible = portfolio_cfg['flexible']
        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if init_position is None:
            init_position = portfolio_cfg['init_position']
        if cash_deposits is None:
            cash_deposits = portfolio_cfg['cash_deposits']
        if cash_earnings is None:
            cash_earnings = portfolio_cfg['cash_earnings']
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']
        if cash_sharing and group_by is None:
            group_by = True
        if not flexible:
            if call_seq is None:
                call_seq = portfolio_cfg['call_seq']
            call_seq = map_enum_fields(call_seq, CallSeqType)
            if isinstance(call_seq, int):
                if call_seq == CallSeqType.Auto:
                    raise ValueError("CallSeqType.Auto must be implemented manually. "
                                     "Use sort_call_seq_nb in pre_segment_func_nb.")
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']
        if segment_mask is None:
            segment_mask = True
        if call_pre_segment is None:
            call_pre_segment = portfolio_cfg['call_pre_segment']
        if call_post_segment is None:
            call_post_segment = portfolio_cfg['call_post_segment']
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']
        if update_value is None:
            update_value = portfolio_cfg['update_value']
        if fill_pos_record is None:
            fill_pos_record = portfolio_cfg['fill_pos_record']
        if track_value is None:
            track_value = portfolio_cfg['track_value']
        if row_wise is None:
            row_wise = portfolio_cfg['row_wise']
        if seed is None:
            seed = portfolio_cfg['seed']
        if seed is not None:
            set_seed(seed)
        if in_outputs is not None and not checks.is_namedtuple(in_outputs):
            in_outputs = to_mapping(in_outputs)
            in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        if group_by is None:
            group_by = portfolio_cfg['group_by']
        if freq is None:
            freq = portfolio_cfg['freq']
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_kwargs = merge_dicts(portfolio_cfg['broadcast_kwargs'], broadcast_kwargs)
        require_kwargs = broadcast_kwargs.get('require_kwargs', {})
        template_mapping = merge_dicts(portfolio_cfg['template_mapping'], template_mapping)
        if keep_inout_raw is None:
            keep_inout_raw = portfolio_cfg['keep_inout_raw']
        if template_mapping is None:
            template_mapping = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")
        if bm_close is None:
            bm_close = portfolio_cfg['bm_close']

        # Prepare the simulation
        broadcastable_args = dict(
            cash_earnings=cash_earnings,
            open=open,
            high=high,
            low=low,
            close=close
        )
        if bm_close is not None and not isinstance(bm_close, bool):
            broadcastable_args['bm_close'] = bm_close
        broadcastable_args = {**broadcastable_args, **broadcast_named_args}
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        broadcast_kwargs = merge_dicts(dict(
            keep_flex=dict(
                close=False,
                bm_close=False,
                _default=True
            )
        ), broadcast_kwargs)
        broadcasted_args = broadcast(broadcastable_args, **broadcast_kwargs)
        cash_earnings = broadcasted_args.pop('cash_earnings')
        close = broadcasted_args['close']
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = broadcasted_args['bm_close']
            if not checks.is_pandas(bm_close):
                bm_close = pd.Series(bm_close) if bm_close.ndim == 1 else pd.DataFrame(bm_close)
            broadcasted_args['bm_close'] = to_2d_array(bm_close)
        else:
            broadcasted_args['bm_close'] = None
        flex_2d = close.ndim == 2
        broadcasted_args['close'] = to_2d_array(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)

        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        init_position = np.require(np.broadcast_to(init_position, (target_shape_2d[1],)), dtype=np.float_)
        cash_deposits = broadcast(
            to_2d_array(cash_deposits, expand_axis=int(not flex_2d)),
            to_shape=(target_shape_2d[0], len(cs_group_lens)),
            to_pd=False,
            keep_flex=keep_inout_raw,
            **require_kwargs
        )
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if isinstance(segment_mask, int):
            if keep_inout_raw:
                _segment_mask = np.full((target_shape_2d[0], 1), False)
            else:
                _segment_mask = np.full((target_shape_2d[0], len(group_lens)), False)
            _segment_mask[0::segment_mask] = True
            segment_mask = _segment_mask
        else:
            segment_mask = broadcast(
                to_2d_array(segment_mask, expand_axis=int(not flex_2d)),
                to_shape=(target_shape_2d[0], len(group_lens)),
                to_pd=False,
                keep_flex=keep_inout_raw,
                **require_kwargs
            )
        if not flexible:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
            else:
                call_seq = build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)

        # Check data types
        checks.assert_subdtype(cs_group_lens, np.int_)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.int_)
        checks.assert_subdtype(init_cash, np.number)
        checks.assert_subdtype(init_position, np.number)
        checks.assert_subdtype(cash_deposits, np.number)
        checks.assert_subdtype(cash_earnings, np.number)
        checks.assert_subdtype(segment_mask, np.bool_)
        checks.assert_subdtype(broadcasted_args['open'], np.number)
        checks.assert_subdtype(broadcasted_args['high'], np.number)
        checks.assert_subdtype(broadcasted_args['low'], np.number)
        checks.assert_subdtype(broadcasted_args['close'], np.number)
        if bm_close is not None and not isinstance(bm_close, bool):
            checks.assert_subdtype(broadcasted_args['bm_close'], np.number)

        # Prepare arguments
        template_mapping = merge_dicts(
            broadcasted_args,
            dict(
                target_shape=target_shape_2d,
                cs_group_lens=cs_group_lens,
                group_lens=group_lens,
                cash_sharing=cash_sharing,
                init_cash=init_cash,
                init_position=init_position,
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
                flex_2d=flex_2d,
                wrapper=wrapper
            ),
            template_mapping
        )
        pre_sim_args = deep_substitute(pre_sim_args, template_mapping, sub_id='pre_sim_args')
        post_sim_args = deep_substitute(post_sim_args, template_mapping, sub_id='post_sim_args')
        pre_group_args = deep_substitute(pre_group_args, template_mapping, sub_id='pre_group_args')
        post_group_args = deep_substitute(post_group_args, template_mapping, sub_id='post_group_args')
        pre_row_args = deep_substitute(pre_row_args, template_mapping, sub_id='pre_row_args')
        post_row_args = deep_substitute(post_row_args, template_mapping, sub_id='post_row_args')
        pre_segment_args = deep_substitute(pre_segment_args, template_mapping, sub_id='pre_segment_args')
        post_segment_args = deep_substitute(post_segment_args, template_mapping, sub_id='post_segment_args')
        order_args = deep_substitute(order_args, template_mapping, sub_id='order_args')
        post_order_args = deep_substitute(post_order_args, template_mapping, sub_id='post_order_args')
        in_outputs = deep_substitute(in_outputs, template_mapping, sub_id='in_outputs')
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
                    open=broadcasted_args['open'],
                    high=broadcasted_args['high'],
                    low=broadcasted_args['low'],
                    close=broadcasted_args['close'],
                    bm_close=broadcasted_args['bm_close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=flex_2d,
                    in_outputs=in_outputs
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
                    open=broadcasted_args['open'],
                    high=broadcasted_args['high'],
                    low=broadcasted_args['low'],
                    close=broadcasted_args['close'],
                    bm_close=broadcasted_args['bm_close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=flex_2d,
                    in_outputs=in_outputs
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
                    open=broadcasted_args['open'],
                    high=broadcasted_args['high'],
                    low=broadcasted_args['low'],
                    close=broadcasted_args['close'],
                    bm_close=broadcasted_args['bm_close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=flex_2d,
                    in_outputs=in_outputs
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
                    open=broadcasted_args['open'],
                    high=broadcasted_args['high'],
                    low=broadcasted_args['low'],
                    close=broadcasted_args['close'],
                    bm_close=broadcasted_args['bm_close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=flex_2d,
                    in_outputs=in_outputs
                )

        # Create an instance
        return cls(
            wrapper,
            close,
            sim_out.order_records,
            sim_out.log_records,
            cash_sharing,
            init_cash if init_cash_mode is None else init_cash_mode,
            init_position=init_position,
            cash_deposits=cash_deposits,
            cash_earnings=sim_out.cash_earnings,
            call_seq=call_seq if not flexible and attach_call_seq else None,
            in_outputs=sim_out.in_outputs,
            bm_close=bm_close,
            **kwargs
        )

    @classmethod
    def from_def_order_func(cls: tp.Type[PortfolioT],
                            close: tp.ArrayLike,
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
                            chunked: tp.ChunkedOption = None,
                            **kwargs) -> PortfolioT:
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
            >>> asset_value.vbt.plot()
            ```

            ![](/assets/images/simulate_nb_example.svg)
        """
        # Get defaults
        from vectorbtpro._settings import settings
        portfolio_cfg = settings['portfolio']

        if flexible is None:
            flexible = portfolio_cfg['flexible']
        if size is None:
            size = portfolio_cfg['size']
        if size_type is None:
            size_type = portfolio_cfg['size_type']
        if direction is None:
            direction = portfolio_cfg['direction']
        if price is None:
            price = portfolio_cfg['price']
        if size is None:
            size = portfolio_cfg['size']
        if fees is None:
            fees = portfolio_cfg['fees']
        if fixed_fees is None:
            fixed_fees = portfolio_cfg['fixed_fees']
        if slippage is None:
            slippage = portfolio_cfg['slippage']
        if min_size is None:
            min_size = portfolio_cfg['min_size']
        if max_size is None:
            max_size = portfolio_cfg['max_size']
        if size_granularity is None:
            size_granularity = portfolio_cfg['size_granularity']
        if reject_prob is None:
            reject_prob = portfolio_cfg['reject_prob']
        if price_area_vio_mode is None:
            price_area_vio_mode = portfolio_cfg['price_area_vio_mode']
        if lock_cash is None:
            lock_cash = portfolio_cfg['lock_cash']
        if allow_partial is None:
            allow_partial = portfolio_cfg['allow_partial']
        if raise_reject is None:
            raise_reject = portfolio_cfg['raise_reject']
        if log is None:
            log = portfolio_cfg['log']
        if val_price is None:
            val_price = portfolio_cfg['val_price']
        if call_seq is None:
            call_seq = portfolio_cfg['call_seq']
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if broadcast_named_args is None:
            broadcast_named_args = {}
        broadcast_named_args = {
            **dict(
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
                val_price=val_price
            ),
            **broadcast_named_args
        }

        # Check data types
        checks.assert_subdtype(size, np.number)
        checks.assert_subdtype(price, np.number)
        checks.assert_subdtype(fees, np.number)
        checks.assert_subdtype(fixed_fees, np.number)
        checks.assert_subdtype(slippage, np.number)
        checks.assert_subdtype(min_size, np.number)
        checks.assert_subdtype(max_size, np.number)
        checks.assert_subdtype(size_granularity, np.number)
        checks.assert_subdtype(reject_prob, np.number)
        checks.assert_subdtype(lock_cash, np.bool_)
        checks.assert_subdtype(allow_partial, np.bool_)
        checks.assert_subdtype(raise_reject, np.bool_)
        checks.assert_subdtype(log, np.bool_)
        checks.assert_subdtype(val_price, np.number)

        def _postprocess_size_type(size_type):
            size_type = map_enum_fields(size_type, SizeType)
            checks.assert_subdtype(size_type, np.int_)
            return size_type

        def _postprocess_direction(direction):
            direction = map_enum_fields(direction, Direction)
            checks.assert_subdtype(direction, np.int_)
            return direction

        def _postprocess_price_area_vio_mode(price_area_vio_mode):
            price_area_vio_mode = map_enum_fields(price_area_vio_mode, PriceAreaVioMode)
            checks.assert_subdtype(price_area_vio_mode, np.int_)
            return price_area_vio_mode

        # Prepare arguments and pass to from_order_func
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
            Rep('size'),
            Rep('price'),
            RepFunc(_postprocess_size_type),
            RepFunc(_postprocess_direction),
            Rep('fees'),
            Rep('fixed_fees'),
            Rep('slippage'),
            Rep('min_size'),
            Rep('max_size'),
            Rep('size_granularity'),
            Rep('reject_prob'),
            RepFunc(_postprocess_price_area_vio_mode),
            Rep('lock_cash'),
            Rep('allow_partial'),
            Rep('raise_reject'),
            Rep('log')
        )
        pre_segment_args = (
            Rep('val_price'),
            Rep('price'),
            Rep('size'),
            RepFunc(_postprocess_size_type),
            RepFunc(_postprocess_direction),
            auto_call_seq
        )
        arg_take_spec = dict(
            pre_segment_args=ch.ArgsTaker(*[
                portfolio_ch.flex_array_gl_slicer if isinstance(x, Rep) else None
                for x in pre_segment_args
            ])
        )
        order_args_taker = ch.ArgsTaker(*[
            portfolio_ch.flex_array_gl_slicer if isinstance(x, Rep) else None
            for x in order_args
        ])
        if flexible:
            arg_take_spec['flex_order_args'] = order_args_taker
        else:
            arg_take_spec['order_args'] = order_args_taker
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=arg_take_spec
        )
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
            **kwargs
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
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Named tuple with in-output objects."""
        return self._in_outputs

    @custom_property(obj_type='array', group_by_aware=False)
    def call_seq(self) -> tp.Optional[tp.SeriesFrame]:
        """Sequence of calls per row and group."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, 'call_seq'):
            call_seq = self.in_outputs.call_seq
        else:
            call_seq = self._call_seq
        if call_seq is None:
            return None

        return self.wrapper.wrap(call_seq, group_by=False)

    # ############# Price ############# #

    @custom_property(obj_type='array', group_by_aware=False)
    def close(self) -> tp.SeriesFrame:
        """Price per unit series."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, 'close'):
            close = self.in_outputs.close
        else:
            close = self._close

        return self.wrapper.wrap(close, group_by=False)

    @class_or_instancemethod
    def get_filled_close(cls_or_self,
                         close: tp.Optional[tp.SeriesFrame] = None,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         wrapper: tp.Optional[ArrayWrapper] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get forward and backward filled closing price.

        See `vectorbtpro.generic.nb.fbfill_nb`."""
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

    @custom_property(obj_type='array', group_by_aware=False)
    def bm_close(self) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Benchmark price per unit series."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, 'bm_close'):
            bm_close = self.in_outputs.bm_close
        else:
            bm_close = self._bm_close

        if bm_close is None or isinstance(bm_close, bool):
            return bm_close
        return self.wrapper.wrap(bm_close, group_by=False)

    @class_or_instancemethod
    def get_filled_bm_close(cls_or_self,
                            bm_close: tp.Optional[tp.SeriesFrame] = None,
                            jitted: tp.JittedOption = None,
                            chunked: tp.ChunkedOption = None,
                            wrapper: tp.Optional[ArrayWrapper] = None,
                            wrap_kwargs: tp.KwargsLike = None) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Get forward and backward filled benchmark closing price.

        See `vectorbtpro.generic.nb.fbfill_nb`."""
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
    def get_orders(cls_or_self,
                   order_records: tp.Optional[tp.RecordArray] = None,
                   close: tp.Optional[tp.SeriesFrame] = None,
                   group_by: tp.GroupByLike = None,
                   wrapper: tp.Optional[ArrayWrapper] = None,
                   **kwargs) -> Orders:
        """Get order records.

        See `vectorbtpro.portfolio.orders.Orders`."""
        if not isinstance(cls_or_self, type):
            if order_records is None:
                order_records = cls_or_self.order_records
            if close is None:
                close = cls_or_self.close
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
        else:
            checks.assert_not_none(order_records)
            checks.assert_not_none(close)
            checks.assert_not_none(wrapper)

        return Orders(wrapper, order_records, close=close, **kwargs).regroup(group_by)

    @property
    def log_records(self) -> tp.RecordArray:
        """A structured NumPy array of log records."""
        return self._log_records

    @class_or_instancemethod
    def get_logs(cls_or_self,
                 log_records: tp.Optional[tp.RecordArray] = None,
                 group_by: tp.GroupByLike = None,
                 wrapper: tp.Optional[ArrayWrapper] = None,
                 **kwargs) -> Logs:
        """Get log records.

        See `vectorbtpro.portfolio.logs.Logs`."""
        if not isinstance(cls_or_self, type):
            if log_records is None:
                log_records = cls_or_self.log_records
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
        else:
            checks.assert_not_none(log_records)
            checks.assert_not_none(wrapper)

        return Logs(wrapper, log_records, **kwargs).regroup(group_by)

    @class_or_instancemethod
    def get_entry_trades(cls_or_self,
                         orders: tp.Optional[Orders] = None,
                         init_position: tp.Optional[tp.ArrayLike] = None,
                         group_by: tp.GroupByLike = None,
                         **kwargs) -> EntryTrades:
        """Get entry trade records.

        See `vectorbtpro.portfolio.trades.EntryTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self.init_position
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.

        return EntryTrades.from_orders(orders, init_position=init_position, **kwargs).regroup(group_by)

    @class_or_instancemethod
    def get_exit_trades(cls_or_self,
                        orders: tp.Optional[Orders] = None,
                        init_position: tp.Optional[tp.ArrayLike] = None,
                        group_by: tp.GroupByLike = None,
                        **kwargs) -> ExitTrades:
        """Get exit trade records.

        See `vectorbtpro.portfolio.trades.ExitTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self.init_position
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.

        return ExitTrades.from_orders(orders, init_position=init_position, **kwargs).regroup(group_by)

    @class_or_instancemethod
    def get_positions(cls_or_self,
                      trades: tp.Optional[Trades] = None,
                      group_by: tp.GroupByLike = None,
                      **kwargs) -> Positions:
        """Get position records.

        See `vectorbtpro.portfolio.trades.Positions`."""
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.exit_trades
        else:
            checks.assert_not_none(trades)

        return Positions.from_trades(trades, **kwargs).regroup(group_by)

    def get_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> Trades:
        """Get trade/position records depending upon `Portfolio.trades_type`."""
        if self.trades_type == TradesType.EntryTrades:
            return self.resolve_shortcut_attr('entry_trades', group_by=group_by, **kwargs)
        elif self.trades_type == TradesType.ExitTrades:
            return self.resolve_shortcut_attr('exit_trades', group_by=group_by, **kwargs)
        return self.resolve_shortcut_attr('positions', group_by=group_by, **kwargs)

    @class_or_instancemethod
    def get_drawdowns(cls_or_self,
                      value: tp.Optional[tp.SeriesFrame] = None,
                      group_by: tp.GroupByLike = None,
                      wrapper_kwargs: tp.KwargsLike = None,
                      **kwargs) -> Drawdowns:
        """Get drawdown records from `Portfolio.get_value`.

        See `vectorbtpro.generic.drawdowns.Drawdowns`."""
        if not isinstance(cls_or_self, type):
            if value is None:
                value = cls_or_self.resolve_shortcut_attr('value', group_by=group_by)
            wrapper_kwargs = merge_dicts(cls_or_self.orders.wrapper.config, wrapper_kwargs, dict(group_by=None))
        else:
            checks.assert_not_none(value)

        return Drawdowns.from_ts(value, wrapper_kwargs=wrapper_kwargs, **kwargs)

    # ############# Assets ############# #

    @class_or_instancemethod
    def get_init_position(cls_or_self,
                          init_position_raw: tp.Optional[tp.ArrayLike] = None,
                          wrapper: tp.Optional[ArrayWrapper] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
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
        wrap_kwargs = merge_dicts(dict(name_or_index='init_position'), wrap_kwargs)
        return wrapper.wrap_reduced(init_position, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_flow(cls_or_self,
                       direction: tp.Union[str, int] = 'both',
                       orders: tp.Optional[Orders] = None,
                       init_position: tp.Optional[tp.ArrayLike] = None,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       wrapper: tp.Optional[ArrayWrapper] = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
                init_position = 0.
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
            init_position=to_1d_array(init_position)
        )
        return wrapper.wrap(asset_flow, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_assets(cls_or_self,
                   direction: tp.Union[str, int] = 'both',
                   asset_flow: tp.Optional[tp.SeriesFrame] = None,
                   init_position: tp.Optional[tp.ArrayLike] = None,
                   jitted: tp.JittedOption = None,
                   chunked: tp.ChunkedOption = None,
                   wrapper: tp.Optional[ArrayWrapper] = None,
                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get asset series per column.

        Returns the position at each time step."""
        if not isinstance(cls_or_self, type):
            if asset_flow is None:
                asset_flow = cls_or_self.resolve_shortcut_attr(
                    'asset_flow',
                    direction=Direction.Both,
                    jitted=jitted,
                    chunked=chunked
                )
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_flow)
            if init_position is None:
                init_position = 0.
            checks.assert_not_none(wrapper)

        direction = map_enum_fields(direction, Direction)
        func = jit_reg.resolve_option(nb.assets_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        assets = func(
            to_2d_array(asset_flow),
            init_position=to_1d_array(init_position)
        )
        if direction == Direction.LongOnly:
            func = jit_reg.resolve_option(nb.longonly_assets_nb, jitted)
            assets = func(assets)
        elif direction == Direction.ShortOnly:
            func = jit_reg.resolve_option(nb.shortonly_assets_nb, jitted)
            assets = func(assets)
        return wrapper.wrap(assets, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_mask(cls_or_self,
                          direction: tp.Union[str, int] = 'both',
                          group_by: tp.GroupByLike = None,
                          assets: tp.Optional[tp.SeriesFrame] = None,
                          jitted: tp.JittedOption = None,
                          chunked: tp.ChunkedOption = None,
                          wrapper: tp.Optional[ArrayWrapper] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get position mask per column/group.

        An element is True if there is a position at the given time step."""
        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    'assets',
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        position_mask = to_2d_array(assets) != 0
        if wrapper.grouper.is_grouped(group_by=group_by):
            position_mask = wrapper.wrap(position_mask, group_by=False) \
                .vbt(wrapper=wrapper) \
                .squeeze_grouped(
                jit_reg.resolve_option(generic_nb.any_reduce_nb, jitted),
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
        return wrapper.wrap(position_mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_coverage(cls_or_self,
                              direction: tp.Union[str, int] = 'both',
                              group_by: tp.GroupByLike = None,
                              position_mask: tp.Optional[tp.SeriesFrame] = None,
                              jitted: tp.JittedOption = None,
                              chunked: tp.ChunkedOption = None,
                              wrapper: tp.Optional[ArrayWrapper] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get position coverage per column/group.

        Position coverage is the number of time steps in the market divided by the total number of time steps."""
        if not isinstance(cls_or_self, type):
            if position_mask is None:
                position_mask = cls_or_self.resolve_shortcut_attr(
                    'position_mask',
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                    group_by=False
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(position_mask)
            checks.assert_not_none(wrapper)

        position_coverage = position_mask \
            .vbt(wrapper=wrapper) \
            .reduce(
            jit_reg.resolve_option(generic_nb.mean_reduce_nb, jitted),
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        wrap_kwargs = merge_dicts(dict(name_or_index='position_coverage'), wrap_kwargs)
        return wrapper.wrap_reduced(position_coverage, group_by=group_by, **wrap_kwargs)

    # ############# Cash ############# #

    @class_or_instancemethod
    def get_init_cash(cls_or_self,
                      group_by: tp.GroupByLike = None,
                      init_cash_raw: tp.Optional[tp.ArrayLike] = None,
                      cash_sharing: tp.Optional[bool] = None,
                      cash_flow: tp.Optional[tp.SeriesFrame] = None,
                      split_shared: bool = False,
                      jitted: tp.JittedOption = None,
                      chunked: tp.ChunkedOption = None,
                      wrapper: tp.Optional[ArrayWrapper] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
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
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        if isinstance(init_cash_raw, int):
            if not isinstance(cls_or_self, type):
                if cash_flow is None:
                    cash_flow = cls_or_self.resolve_shortcut_attr(
                        'cash_flow',
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked
                    )
            func = jit_reg.resolve_option(nb.align_init_cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            init_cash = func(init_cash_raw, to_2d_array(cash_flow))
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
        wrap_kwargs = merge_dicts(dict(name_or_index='init_cash'), wrap_kwargs)
        return wrapper.wrap_reduced(init_cash, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_cash_deposits(cls_or_self,
                          group_by: tp.GroupByLike = None,
                          cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
                          cash_sharing: tp.Optional[bool] = None,
                          split_shared: bool = False,
                          flex_2d: bool = False,
                          keep_flex: bool = False,
                          jitted: tp.JittedOption = None,
                          chunked: tp.ChunkedOption = None,
                          wrapper: tp.Optional[ArrayWrapper] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.ArrayLike:
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
                cash_deposits_raw = 0.
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(wrapper)

        cash_deposits_raw = to_2d_array(cash_deposits_raw)
        if wrapper.grouper.is_grouped(group_by=group_by):
            if keep_flex and cash_sharing:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_deposits_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(
                wrapper.shape_2d,
                cash_deposits_raw,
                group_lens,
                cash_sharing,
                flex_2d=flex_2d
            )
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
                flex_2d=flex_2d
            )
        if keep_flex:
            return cash_deposits
        return wrapper.wrap(cash_deposits, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash_earnings(cls_or_self,
                          group_by: tp.GroupByLike = None,
                          cash_earnings_raw: tp.Optional[tp.ArrayLike] = None,
                          keep_flex: bool = False,
                          jitted: tp.JittedOption = None,
                          chunked: tp.ChunkedOption = None,
                          wrapper: tp.Optional[ArrayWrapper] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.ArrayLike:
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
                cash_earnings_raw = 0.
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
    def get_cash_flow(cls_or_self,
                      group_by: tp.GroupByLike = None,
                      free: bool = False,
                      orders: tp.Optional[Orders] = None,
                      cash_earnings: tp.Optional[tp.ArrayLike] = None,
                      flex_2d: bool = False,
                      jitted: tp.JittedOption = None,
                      chunked: tp.ChunkedOption = None,
                      wrapper: tp.Optional[ArrayWrapper] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
                cash_earnings = 0.
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
            flex_2d=flex_2d
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_flow = func(cash_flow, group_lens)
        return wrapper.wrap(cash_flow, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash(cls_or_self,
                 group_by: tp.GroupByLike = None,
                 free: bool = False,
                 cash_sharing: tp.Optional[bool] = None,
                 init_cash: tp.Optional[tp.ArrayLike] = None,
                 cash_deposits: tp.Optional[tp.ArrayLike] = None,
                 cash_flow: tp.Optional[tp.SeriesFrame] = None,
                 flex_2d: bool = False,
                 jitted: tp.JittedOption = None,
                 chunked: tp.ChunkedOption = None,
                 wrapper: tp.Optional[ArrayWrapper] = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get cash balance series per column/group.

        For `free`, see `Portfolio.get_cash_flow`."""
        if not isinstance(cls_or_self, type):
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    'cash_flow',
                    group_by=group_by,
                    free=free,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(init_cash)
            if cash_deposits is None:
                cash_deposits = 0.
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        'init_cash',
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        'cash_deposits',
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True
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
                flex_2d=flex_2d
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        'init_cash',
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        'cash_deposits',
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True
                    )
            func = jit_reg.resolve_option(nb.cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash = func(
                to_2d_array(cash_flow),
                to_1d_array(init_cash),
                cash_deposits=to_2d_array(cash_deposits),
                flex_2d=flex_2d
            )
        return wrapper.wrap(cash, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Value ############# #

    @class_or_instancemethod
    def get_init_position_value(cls_or_self,
                                close: tp.Optional[tp.SeriesFrame] = None,
                                init_position: tp.Optional[tp.ArrayLike] = None,
                                jitted: tp.JittedOption = None,
                                chunked: tp.ChunkedOption = None,
                                wrapper: tp.Optional[ArrayWrapper] = None,
                                wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get initial position value per column."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            if init_position is None:
                init_position = 0.
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.init_position_value_nb, jitted)
        init_position_value = func(to_2d_array(close), init_position=to_1d_array(init_position))
        wrap_kwargs = merge_dicts(dict(name_or_index='init_position_value'), wrap_kwargs)
        return wrapper.wrap_reduced(init_position_value, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_init_value(cls_or_self,
                       group_by: tp.GroupByLike = None,
                       init_position_value: tp.Optional[tp.MaybeSeries] = None,
                       init_cash: tp.Optional[tp.MaybeSeries] = None,
                       split_shared: bool = False,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       wrapper: tp.Optional[ArrayWrapper] = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get initial value per column/group.

        Includes initial cash and the value of initial position."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    'init_cash',
                    group_by=group_by,
                    split_shared=split_shared,
                    jitted=jitted,
                    chunked=chunked
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
            init_value = func(
                group_lens,
                to_1d_array(init_position_value),
                to_1d_array(init_cash)
            )
        else:
            func = jit_reg.resolve_option(nb.init_value_nb, jitted)
            init_value = func(
                to_1d_array(init_position_value),
                to_1d_array(init_cash)
            )
        wrap_kwargs = merge_dicts(dict(name_or_index='init_value'), wrap_kwargs)
        return wrapper.wrap_reduced(init_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_input_value(cls_or_self,
                        group_by: tp.GroupByLike = None,
                        cash_sharing: tp.Optional[bool] = None,
                        init_value: tp.Optional[tp.MaybeSeries] = None,
                        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
                        split_shared: bool = False,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        wrapper: tp.Optional[ArrayWrapper] = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get total input value per column/group.

        Includes initial value and any cash deposited at any point in time."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    'init_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
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
                cash_deposits_raw = 0.
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
        wrap_kwargs = merge_dicts(dict(name_or_index='input_value'), wrap_kwargs)
        return wrapper.wrap_reduced(input_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_value(cls_or_self,
                        direction: tp.Union[str, int] = 'both',
                        group_by: tp.GroupByLike = None,
                        close: tp.Optional[tp.SeriesFrame] = None,
                        assets: tp.Optional[tp.SeriesFrame] = None,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        wrapper: tp.Optional[ArrayWrapper] = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get asset value series per column/group."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    'assets',
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        close = to_2d_array(close).copy()
        assets = to_2d_array(assets)
        close[assets == 0] = 0.  # for price being NaN
        func = jit_reg.resolve_option(nb.asset_value_nb, jitted)
        asset_value = func(close, assets)
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            asset_value = func(asset_value, group_lens)
        return wrapper.wrap(asset_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_gross_exposure(cls_or_self,
                           direction: tp.Union[str, int] = 'both',
                           group_by: tp.GroupByLike = None,
                           asset_value: tp.Optional[tp.SeriesFrame] = None,
                           free_cash: tp.Optional[tp.SeriesFrame] = None,
                           jitted: tp.JittedOption = None,
                           chunked: tp.ChunkedOption = None,
                           wrapper: tp.Optional[ArrayWrapper] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get gross exposure."""
        direction = map_enum_fields(direction, Direction)

        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    'asset_value',
                    group_by=group_by,
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked
                )
            if free_cash is None:
                free_cash = cls_or_self.resolve_shortcut_attr(
                    'cash',
                    group_by=group_by,
                    free=True,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value)
            checks.assert_not_none(free_cash)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.gross_exposure_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        gross_exposure = func(to_2d_array(asset_value), to_2d_array(free_cash))
        return wrapper.wrap(gross_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_net_exposure(cls_or_self,
                         group_by: tp.GroupByLike = None,
                         long_exposure: tp.Optional[tp.SeriesFrame] = None,
                         short_exposure: tp.Optional[tp.SeriesFrame] = None,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         wrapper: tp.Optional[ArrayWrapper] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get net exposure."""
        if not isinstance(cls_or_self, type):
            if long_exposure is None:
                long_exposure = cls_or_self.resolve_shortcut_attr(
                    'gross_exposure',
                    direction=Direction.LongOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if short_exposure is None:
                short_exposure = cls_or_self.resolve_shortcut_attr(
                    'gross_exposure',
                    direction=Direction.ShortOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
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
    def get_value(cls_or_self,
                  group_by: tp.GroupByLike = None,
                  cash: tp.Optional[tp.SeriesFrame] = None,
                  asset_value: tp.Optional[tp.SeriesFrame] = None,
                  jitted: tp.JittedOption = None,
                  chunked: tp.ChunkedOption = None,
                  wrapper: tp.Optional[ArrayWrapper] = None,
                  wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get portfolio value series per column/group.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with the initial cash balance and position being that of the
        entire group. Useful for generating returns and comparing assets within the same group."""
        if not isinstance(cls_or_self, type):
            if cash is None:
                cash = cls_or_self.resolve_shortcut_attr(
                    'cash',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    'asset_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.value_nb, jitted)
        value = func(to_2d_array(cash), to_2d_array(asset_value))
        return wrapper.wrap(value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_total_profit(cls_or_self,
                         group_by: tp.GroupByLike = None,
                         close: tp.Optional[tp.SeriesFrame] = None,
                         orders: tp.Optional[Orders] = None,
                         init_position: tp.Optional[tp.ArrayLike] = None,
                         cash_earnings: tp.Optional[tp.ArrayLike] = None,
                         flex_2d: bool = False,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         wrapper: tp.Optional[ArrayWrapper] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
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
            if cash_earnings is None:
                cash_earnings = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if close is None:
                close = orders.close
            checks.assert_not_none(close)
            if init_position is None:
                init_position = 0.
            if cash_earnings is None:
                cash_earnings = 0.
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
            cash_earnings=to_2d_array(cash_earnings),
            flex_2d=flex_2d
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.total_profit_grouped_nb, jitted)
            total_profit = func(total_profit, group_lens)
        wrap_kwargs = merge_dicts(dict(name_or_index='total_profit'), wrap_kwargs)
        return wrapper.wrap_reduced(total_profit, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_final_value(cls_or_self,
                        group_by: tp.GroupByLike = None,
                        input_value: tp.Optional[tp.MaybeSeries] = None,
                        total_profit: tp.Optional[tp.MaybeSeries] = None,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        wrapper: tp.Optional[ArrayWrapper] = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get total profit per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    'input_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    'total_profit',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        final_value = to_1d_array(input_value) + to_1d_array(total_profit)
        wrap_kwargs = merge_dicts(dict(name_or_index='final_value'), wrap_kwargs)
        return wrapper.wrap_reduced(final_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_total_return(cls_or_self,
                         group_by: tp.GroupByLike = None,
                         input_value: tp.Optional[tp.MaybeSeries] = None,
                         total_profit: tp.Optional[tp.MaybeSeries] = None,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         wrapper: tp.Optional[ArrayWrapper] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get total return per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    'input_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    'total_profit',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        total_return = to_1d_array(total_profit) / to_1d_array(input_value)
        wrap_kwargs = merge_dicts(dict(name_or_index='total_return'), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns(cls_or_self,
                    group_by: tp.GroupByLike = None,
                    init_value: tp.Optional[tp.MaybeSeries] = None,
                    cash_deposits: tp.Optional[tp.ArrayLike] = None,
                    value: tp.Optional[tp.SeriesFrame] = None,
                    flex_2d: bool = False,
                    jitted: tp.JittedOption = None,
                    chunked: tp.ChunkedOption = None,
                    wrapper: tp.Optional[ArrayWrapper] = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get return series per column/group based on portfolio value."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    'init_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    'cash_deposits',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    'value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        returns = func(
            to_2d_array(value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            flex_2d=flex_2d
        )
        return wrapper.wrap(returns, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_asset_returns(cls_or_self,
                          group_by: tp.GroupByLike = None,
                          init_position_value: tp.Optional[tp.MaybeSeries] = None,
                          asset_value: tp.Optional[tp.SeriesFrame] = None,
                          cash_flow: tp.Optional[tp.SeriesFrame] = None,
                          jitted: tp.JittedOption = None,
                          chunked: tp.ChunkedOption = None,
                          wrapper: tp.Optional[ArrayWrapper] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
                    'asset_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    'cash_flow',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
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
            to_2d_array(cash_flow)
        )
        return wrapper.wrap(asset_returns, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_market_value(cls_or_self,
                         group_by: tp.GroupByLike = None,
                         close: tp.Optional[tp.SeriesFrame] = None,
                         init_value: tp.Optional[tp.MaybeSeries] = None,
                         cash_deposits: tp.Optional[tp.ArrayLike] = None,
                         flex_2d: bool = False,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         wrapper: tp.Optional[ArrayWrapper] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
                cash_deposits = 0.
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        'init_value',
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        'cash_deposits',
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True
                    )
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.market_value_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                group_lens,
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
                flex_2d=flex_2d
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        'init_value',
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        'cash_deposits',
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True
                    )
            func = jit_reg.resolve_option(nb.market_value_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
                flex_2d=flex_2d
            )
        return wrapper.wrap(market_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_market_returns(cls_or_self,
                           group_by: tp.GroupByLike = None,
                           init_value: tp.Optional[tp.MaybeSeries] = None,
                           cash_deposits: tp.Optional[tp.ArrayLike] = None,
                           market_value: tp.Optional[tp.SeriesFrame] = None,
                           flex_2d: bool = False,
                           jitted: tp.JittedOption = None,
                           chunked: tp.ChunkedOption = None,
                           wrapper: tp.Optional[ArrayWrapper] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get market return series per column/group."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    'init_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    'cash_deposits',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    'market_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.
            checks.assert_not_none(market_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        market_returns = func(
            to_2d_array(market_value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            flex_2d=flex_2d
        )
        return wrapper.wrap(market_returns, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_bm_value(cls_or_self,
                     group_by: tp.GroupByLike = None,
                     bm_close: tp.Optional[tp.ArrayLike] = None,
                     init_value: tp.Optional[tp.MaybeSeries] = None,
                     cash_deposits: tp.Optional[tp.ArrayLike] = None,
                     flex_2d: bool = False,
                     jitted: tp.JittedOption = None,
                     chunked: tp.ChunkedOption = None,
                     wrapper: tp.Optional[ArrayWrapper] = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.Optional[tp.SeriesFrame]:
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
            flex_2d=flex_2d,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs
        )

    @class_or_instancemethod
    def get_bm_returns(cls_or_self,
                       group_by: tp.GroupByLike = None,
                       init_value: tp.Optional[tp.MaybeSeries] = None,
                       cash_deposits: tp.Optional[tp.ArrayLike] = None,
                       bm_value: tp.Optional[tp.SeriesFrame] = None,
                       flex_2d: bool = False,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       wrapper: tp.Optional[ArrayWrapper] = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark return series per column/group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_returns`."""
        if not isinstance(cls_or_self, type):
            bm_value = cls_or_self.resolve_shortcut_attr(
                'bm_value',
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
            if bm_value is None:
                return None
        return cls_or_self.get_market_returns(
            group_by=group_by,
            init_value=init_value,
            cash_deposits=cash_deposits,
            market_value=bm_value,
            flex_2d=flex_2d,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs
        )

    @class_or_instancemethod
    def get_total_market_return(cls_or_self,
                                group_by: tp.GroupByLike = None,
                                input_value: tp.Optional[tp.MaybeSeries] = None,
                                market_value: tp.Optional[tp.SeriesFrame] = None,
                                jitted: tp.JittedOption = None,
                                chunked: tp.ChunkedOption = None,
                                wrapper: tp.Optional[ArrayWrapper] = None,
                                wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get total market return."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    'input_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    'market_value',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
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
        wrap_kwargs = merge_dicts(dict(name_or_index='total_market_return'), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns_acc(cls_or_self,
                        group_by: tp.GroupByLike = None,
                        returns: tp.Optional[tp.SeriesFrame] = None,
                        bm_returns: tp.Optional[tp.ArrayLike] = None,
                        freq: tp.Optional[tp.FrequencyLike] = None,
                        year_freq: tp.Optional[tp.FrequencyLike] = None,
                        use_asset_returns: bool = False,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        defaults: tp.KwargsLike = None,
                        **kwargs) -> ReturnsAccessor:
        """Get returns accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`.

        !!! hint
            You can find most methods of this accessor as (cacheable) attributes of this portfolio."""
        if not isinstance(cls_or_self, type):
            if returns is None:
                if use_asset_returns:
                    returns = cls_or_self.resolve_shortcut_attr(
                        'asset_returns',
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked
                    )
                else:
                    returns = cls_or_self.resolve_shortcut_attr(
                        'returns',
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked
                    )
            if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
                bm_returns = cls_or_self.resolve_shortcut_attr(
                    'bm_returns',
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked
                )
            elif isinstance(bm_returns, bool) and not bm_returns:
                bm_returns = None
            if freq is None:
                freq = cls_or_self.wrapper.freq
        else:
            checks.assert_not_none(returns)

        return returns.vbt.returns(
            bm_returns=bm_returns,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            **kwargs
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """`Portfolio.get_returns_acc` with default arguments."""
        return self.get_returns_acc()

    @class_or_instancemethod
    def get_qs(cls_or_self,
               group_by: tp.GroupByLike = None,
               returns: tp.Optional[tp.SeriesFrame] = None,
               bm_returns: tp.Optional[tp.ArrayLike] = None,
               freq: tp.Optional[tp.FrequencyLike] = None,
               year_freq: tp.Optional[tp.FrequencyLike] = None,
               use_asset_returns: bool = False,
               jitted: tp.JittedOption = None,
               chunked: tp.ChunkedOption = None,
               defaults: tp.KwargsLike = None,
               **kwargs) -> QSAdapterT:
        """Get quantstats adapter of type `vectorbtpro.returns.qs_adapter.QSAdapter`.

        `**kwargs` are passed to the adapter constructor."""
        from vectorbtpro.returns.qs_adapter import QSAdapter

        returns_acc = cls_or_self.get_returns_acc(
            group_by=group_by,
            returns=returns,
            bm_returns=bm_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            jitted=jitted,
            chunked=chunked,
            defaults=defaults
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
        return {'self', 'portfolio', 'pf'}

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Uses the following keys:

        * `use_asset_returns`: Whether to use `Portfolio.get_asset_returns` when resolving `returns` argument.
        * `trades_type`: Which trade type to use when resolving `trades` argument."""
        if 'use_asset_returns' in final_kwargs:
            if attr == 'returns' and final_kwargs['use_asset_returns']:
                attr = 'asset_returns'
        if 'trades_type' in final_kwargs:
            trades_type = final_kwargs['trades_type']
            if isinstance(final_kwargs['trades_type'], str):
                trades_type = map_enum_fields(trades_type, TradesType)
            if attr == 'trades' and trades_type != self.trades_type:
                if trades_type == TradesType.EntryTrades:
                    attr = 'entry_trades'
                elif trades_type == TradesType.ExitTrades:
                    attr = 'exit_trades'
                else:
                    attr = 'positions'
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process an object after resolution.

        Uses the following keys:

        * `incl_open`: Whether to include open trades/positions when resolving an argument
            that is an instance of `vectorbtpro.portfolio.trades.Trades`."""
        if 'incl_open' in final_kwargs:
            if isinstance(out, Trades) and not final_kwargs['incl_open']:
                out = out.closed
        return out

    def resolve_shortcut_attr(self, attr_name: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute that may have shortcut properties.

        If `attr_name` has a prefix `get_`, checks whether the respective shortcut property can be called.
        This way, complex call hierarchies can utilize cacheable properties."""
        if not attr_name.startswith('get_'):
            if 'get_' + attr_name not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr_name), property):
                    return getattr(self, attr_name)
                return getattr(self, attr_name)(*args, **kwargs)
            attr_name = 'get_' + attr_name

        if len(args) == 0:
            naked_attr_name = attr_name[4:]
            prop_name = naked_attr_name
            _kwargs = dict(kwargs)

            if 'free' in _kwargs:
                if _kwargs.pop('free'):
                    prop_name = 'free_' + naked_attr_name
            if 'direction' in _kwargs:
                direction = map_enum_fields(_kwargs.pop('direction'), Direction)
                if direction == Direction.LongOnly:
                    prop_name = 'longonly_' + naked_attr_name
                elif direction == Direction.ShortOnly:
                    prop_name = 'shortonly_' + naked_attr_name

            if prop_name in self.cls_dir:
                prop = getattr(type(self), prop_name)
                options = getattr(prop, 'options', {})

                can_call_prop = True
                if 'group_by' in _kwargs:
                    group_by = _kwargs.pop('group_by')
                    group_aware = options.get('group_aware', True)
                    if group_aware:
                        if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                            can_call_prop = False
                    else:
                        group_by = _kwargs.pop('group_by')
                        if self.wrapper.grouper.is_grouping_enabled(group_by=group_by):
                            can_call_prop = False
                if can_call_prop:
                    _kwargs.pop('jitted', None)
                    _kwargs.pop('chunked', None)
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
        returns_cfg = settings['returns']
        portfolio_stats_cfg = settings['portfolio']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            dict(
                settings=dict(
                    year_freq=returns_cfg['year_freq'],
                    trades_type=self.trades_type
                )
            ),
            portfolio_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags='wrapper'
            ),
            start_value=dict(
                title='Start Value',
                calc_func='init_cash',
                tags='portfolio'
            ),
            end_value=dict(
                title='End Value',
                calc_func='final_value',
                tags='portfolio'
            ),
            total_return=dict(
                title='Total Return [%]',
                calc_func='total_return',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='portfolio'
            ),
            bm_return=dict(
                title='Benchmark Return [%]',
                calc_func='bm_returns.vbt.returns.total',
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_bm_returns=True,
                tags='portfolio'
            ),
            max_gross_exposure=dict(
                title='Max Gross Exposure [%]',
                calc_func='gross_exposure.vbt.max',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='portfolio'
            ),
            total_fees_paid=dict(
                title='Total Fees Paid',
                calc_func='orders.fees.sum',
                tags=['portfolio', 'orders']
            ),
            max_dd=dict(
                title='Max Drawdown [%]',
                calc_func='drawdowns.max_drawdown',
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=['portfolio', 'drawdowns']
            ),
            max_dd_duration=dict(
                title='Max Drawdown Duration',
                calc_func='drawdowns.max_duration',
                fill_wrap_kwargs=True,
                tags=['portfolio', 'drawdowns', 'duration']
            ),
            total_trades=dict(
                title='Total Trades',
                calc_func='trades.count',
                incl_open=True,
                tags=['portfolio', 'trades']
            ),
            total_closed_trades=dict(
                title='Total Closed Trades',
                calc_func='trades.closed.count',
                tags=['portfolio', 'trades', 'closed']
            ),
            total_open_trades=dict(
                title='Total Open Trades',
                calc_func='trades.open.count',
                incl_open=True,
                tags=['portfolio', 'trades', 'open']
            ),
            open_trade_pnl=dict(
                title='Open Trade PnL',
                calc_func='trades.open.pnl.sum',
                incl_open=True,
                tags=['portfolio', 'trades', 'open']
            ),
            win_rate=dict(
                title='Win Rate [%]',
                calc_func='trades.win_rate',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            best_trade=dict(
                title='Best Trade [%]',
                calc_func='trades.returns.max',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            worst_trade=dict(
                title='Worst Trade [%]',
                calc_func='trades.returns.min',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            avg_winning_trade=dict(
                title='Avg Winning Trade [%]',
                calc_func='trades.winning.returns.mean',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning']")
            ),
            avg_losing_trade=dict(
                title='Avg Losing Trade [%]',
                calc_func='trades.losing.returns.mean',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing']")
            ),
            avg_winning_trade_duration=dict(
                title='Avg Winning Trade Duration',
                calc_func='trades.winning.duration.mean',
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning', 'duration']")
            ),
            avg_losing_trade_duration=dict(
                title='Avg Losing Trade Duration',
                calc_func='trades.losing.duration.mean',
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing', 'duration']")
            ),
            profit_factor=dict(
                title='Profit Factor',
                calc_func='trades.profit_factor',
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            expectancy=dict(
                title='Expectancy',
                calc_func='trades.expectancy',
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            sharpe_ratio=dict(
                title='Sharpe Ratio',
                calc_func='returns_acc.sharpe_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            calmar_ratio=dict(
                title='Calmar Ratio',
                calc_func='returns_acc.calmar_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            omega_ratio=dict(
                title='Omega Ratio',
                calc_func='returns_acc.omega_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            sortino_ratio=dict(
                title='Sortino Ratio',
                calc_func='returns_acc.sortino_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            )
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    def returns_stats(self,
                      group_by: tp.GroupByLike = None,
                      bm_returns: tp.Optional[tp.ArrayLike] = None,
                      freq: tp.Optional[tp.FrequencyLike] = None,
                      year_freq: tp.Optional[tp.FrequencyLike] = None,
                      use_asset_returns: bool = False,
                      defaults: tp.KwargsLike = None,
                      chunked: tp.ChunkedOption = None,
                      **kwargs) -> tp.SeriesFrame:
        """Compute various statistics on returns of this portfolio.

        See `Portfolio.returns_acc` and `vectorbtpro.returns.accessors.ReturnsAccessor.metrics`.

        `kwargs` will be passed to `vectorbtpro.returns.accessors.ReturnsAccessor.stats` method.
        If `bm_returns` is not set, uses `Portfolio.get_market_returns`."""
        returns_acc = self.get_returns_acc(
            group_by=group_by,
            bm_returns=bm_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            defaults=defaults,
            chunked=chunked
        )
        return getattr(returns_acc, 'stats')(**kwargs)

    # ############# Plotting ############# #

    def plot_orders(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of orders."""
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.orders.regroup(False).plot(column=column, **kwargs)

    def plot_trades(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of trades."""
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.trades.regroup(False).plot(column=column, **kwargs)

    def plot_trade_pnl(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of trade PnL."""
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.trades.regroup(False).plot_pnl(column=column, **kwargs)

    def plot_positions(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of positions."""
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.positions.regroup(False).plot(column=column, **kwargs)

    def plot_position_pnl(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of position PnL."""
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.positions.regroup(False).plot_pnl(column=column, **kwargs)

    def plot_asset_flow(self,
                        column: tp.Optional[tp.Label] = None,
                        direction: tp.Union[str, int] = 'both',
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        xref: str = 'x',
                        yref: str = 'y',
                        hline_shape_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column of asset flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        asset_flow = self.resolve_shortcut_attr(
            'asset_flow',
            direction=direction,
            jitted=jitted,
            chunked=chunked
        )
        asset_flow = self.select_one_from_obj(asset_flow, self.wrapper.regroup(False), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['brown']
                ),
                name='Assets'
            )
        ), kwargs)
        fig = asset_flow.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        return fig

    def plot_cash_flow(self,
                       column: tp.Optional[tp.Label] = None,
                       group_by: tp.GroupByLike = None,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       free: bool = False,
                       xref: str = 'x',
                       yref: str = 'y',
                       hline_shape_kwargs: tp.KwargsLike = None,
                       **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of cash flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        cash_flow = self.resolve_shortcut_attr(
            'cash_flow',
            group_by=group_by,
            free=free,
            jitted=jitted,
            chunked=chunked
        )
        cash_flow = self.select_one_from_obj(cash_flow, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['green']
                ),
                name='Cash'
            )
        ), kwargs)
        fig = cash_flow.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_assets(self,
                    column: tp.Optional[tp.Label] = None,
                    direction: tp.Union[str, int] = 'both',
                    jitted: tp.JittedOption = None,
                    chunked: tp.ChunkedOption = None,
                    xref: str = 'x',
                    yref: str = 'y',
                    hline_shape_kwargs: tp.KwargsLike = None,
                    **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column of assets.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        assets = self.resolve_shortcut_attr(
            'assets',
            direction=direction,
            jitted=jitted,
            chunked=chunked
        )
        assets = self.select_one_from_obj(assets, self.wrapper.regroup(False), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['brown']
                ),
                name='Assets'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['brown'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = assets.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_cash(self,
                  column: tp.Optional[tp.Label] = None,
                  group_by: tp.GroupByLike = None,
                  jitted: tp.JittedOption = None,
                  chunked: tp.ChunkedOption = None,
                  free: bool = False,
                  xref: str = 'x',
                  yref: str = 'y',
                  hline_shape_kwargs: tp.KwargsLike = None,
                  **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of cash balance.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        init_cash = self.resolve_shortcut_attr(
            'init_cash',
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        init_cash = self.select_one_from_obj(init_cash, self.wrapper.regroup(group_by), column=column)
        cash = self.resolve_shortcut_attr(
            'cash',
            group_by=group_by,
            free=free,
            jitted=jitted,
            chunked=chunked
        )
        cash = self.select_one_from_obj(cash, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['green']
                ),
                name='Cash'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['green'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['red'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = cash.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=init_cash,
            x1=x_domain[1],
            y1=init_cash
        ), hline_shape_kwargs))
        return fig

    def plot_asset_value(self,
                         column: tp.Optional[tp.Label] = None,
                         group_by: tp.GroupByLike = None,
                         direction: tp.Union[str, int] = 'both',
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         xref: str = 'x',
                         yref: str = 'y',
                         hline_shape_kwargs: tp.KwargsLike = None,
                         **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of asset value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        asset_value = self.resolve_shortcut_attr(
            'asset_value',
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        asset_value = self.select_one_from_obj(asset_value, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['cyan']
                ),
                name='Asset Value'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['cyan'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = asset_value.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_value(self,
                   column: tp.Optional[tp.Label] = None,
                   group_by: tp.GroupByLike = None,
                   jitted: tp.JittedOption = None,
                   chunked: tp.ChunkedOption = None,
                   xref: str = 'x',
                   yref: str = 'y',
                   hline_shape_kwargs: tp.KwargsLike = None,
                   **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        init_cash = self.resolve_shortcut_attr(
            'init_cash',
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        init_cash = self.select_one_from_obj(init_cash, self.wrapper.regroup(group_by), column=column)
        value = self.resolve_shortcut_attr(
            'value',
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        value = self.select_one_from_obj(value, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                ),
                name='Value'
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = value.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=init_cash,
            x1=x_domain[1],
            y1=init_cash
        ), hline_shape_kwargs))
        return fig

    def plot_cum_returns(self,
                         column: tp.Optional[tp.Label] = None,
                         group_by: tp.GroupByLike = None,
                         bm_returns: tp.Optional[tp.ArrayLike] = None,
                         use_asset_returns: bool = False,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of cumulative returns.

        If `bm_returns` is None, will use `Portfolio.get_market_returns`.

        `**kwargs` are passed to `vectorbtpro.returns.accessors.ReturnsSRAccessor.plot_cumulative`."""
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
            bm_returns = self.resolve_shortcut_attr(
                'bm_returns',
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
        elif isinstance(bm_returns, bool) and not bm_returns:
            bm_returns = None
        else:
            bm_returns = broadcast_to(bm_returns, self.obj)
        bm_returns = self.select_one_from_obj(bm_returns, self.wrapper.regroup(group_by), column=column)
        if use_asset_returns:
            returns = self.resolve_shortcut_attr(
                'asset_returns',
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
        else:
            returns = self.resolve_shortcut_attr(
                'returns',
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
        returns = self.select_one_from_obj(returns, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            bm_returns=bm_returns,
            main_kwargs=dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg['color_schema']['purple']
                    ),
                    name='Value'
                )
            ),
            hline_shape_kwargs=dict(
                type='line',
                line=dict(
                    color='gray',
                    dash="dash",
                )
            )
        ), kwargs)
        return returns.vbt.returns.plot_cumulative(**kwargs)

    def plot_drawdowns(self,
                       column: tp.Optional[tp.Label] = None,
                       group_by: tp.GroupByLike = None,
                       **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of drawdowns.

        `**kwargs` are passed to `vectorbtpro.generic.drawdowns.Drawdowns.plot`."""
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            ts_trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                ),
                name='Value'
            )
        ), kwargs)
        return self.resolve_shortcut_attr(
            'drawdowns',
            group_by=group_by
        ).plot(column=column, **kwargs)

    def plot_underwater(self,
                        column: tp.Optional[tp.Label] = None,
                        group_by: tp.GroupByLike = None,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        xref: str = 'x',
                        yref: str = 'y',
                        hline_shape_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of underwater.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        drawdown = self.resolve_shortcut_attr(
            'drawdown',
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        drawdown = self.select_one_from_obj(drawdown, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['red']
                ),
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['red'], 0.3),
                fill='tozeroy',
                name='Drawdown'
            )
        ), kwargs)
        fig = drawdown.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        yaxis = 'yaxis' + yref[1:]
        fig.layout[yaxis]['tickformat'] = '%'
        return fig

    def plot_gross_exposure(self,
                            column: tp.Optional[tp.Label] = None,
                            group_by: tp.GroupByLike = None,
                            direction: tp.Union[str, int] = 'both',
                            jitted: tp.JittedOption = None,
                            chunked: tp.ChunkedOption = None,
                            xref: str = 'x',
                            yref: str = 'y',
                            hline_shape_kwargs: tp.KwargsLike = None,
                            **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of gross exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        gross_exposure = self.resolve_shortcut_attr(
            'gross_exposure',
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        gross_exposure = self.select_one_from_obj(gross_exposure, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['pink']
                ),
                name='Exposure'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['pink'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = gross_exposure.vbt.plot_against(1, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=1,
            x1=x_domain[1],
            y1=1
        ), hline_shape_kwargs))
        return fig

    def plot_net_exposure(self,
                          column: tp.Optional[tp.Label] = None,
                          group_by: tp.GroupByLike = None,
                          jitted: tp.JittedOption = None,
                          chunked: tp.ChunkedOption = None,
                          xref: str = 'x',
                          yref: str = 'y',
                          hline_shape_kwargs: tp.KwargsLike = None,
                          **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot one column/group of net exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings
        plotting_cfg = settings['plotting']

        net_exposure = self.resolve_shortcut_attr(
            'net_exposure',
            group_by=group_by,
            jitted=jitted,
            chunked=chunked
        )
        net_exposure = self.select_one_from_obj(net_exposure, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['pink']
                ),
                name='Exposure'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['pink'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        fig = net_exposure.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.plot`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.portfolio`."""
        from vectorbtpro._settings import settings
        returns_cfg = settings['returns']
        portfolio_plots_cfg = settings['portfolio']['plots']

        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),
            dict(
                settings=dict(
                    year_freq=returns_cfg['year_freq'],
                    trades_type=self.trades_type
                )
            ),
            portfolio_plots_cfg
        )

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            orders=dict(
                title="Orders",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func='orders.plot',
                tags=['portfolio', 'orders']
            ),
            trades=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func='trades.plot',
                tags=['portfolio', 'trades']
            ),
            trade_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func='trades.plot_pnl',
                tags=['portfolio', 'trades']
            ),
            asset_flow=dict(
                title="Asset Flow",
                yaxis_kwargs=dict(title="Asset flow"),
                check_is_not_grouped=True,
                plot_func='plot_asset_flow',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'assets']
            ),
            cash_flow=dict(
                title="Cash Flow",
                yaxis_kwargs=dict(title="Cash flow"),
                plot_func='plot_cash_flow',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'cash']
            ),
            assets=dict(
                title="Assets",
                yaxis_kwargs=dict(title="Assets"),
                check_is_not_grouped=True,
                plot_func='plot_assets',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'assets']
            ),
            cash=dict(
                title="Cash",
                yaxis_kwargs=dict(title="Cash"),
                plot_func='plot_cash',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'cash']
            ),
            asset_value=dict(
                title="Asset Value",
                yaxis_kwargs=dict(title="Asset value"),
                plot_func='plot_asset_value',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'assets', 'value']
            ),
            value=dict(
                title="Value",
                yaxis_kwargs=dict(title="Value"),
                plot_func='plot_value',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'value']
            ),
            cum_returns=dict(
                title="Cumulative Returns",
                yaxis_kwargs=dict(title="Cumulative returns"),
                plot_func='plot_cum_returns',
                pass_hline_shape_kwargs=True,
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=['portfolio', 'returns']
            ),
            drawdowns=dict(
                title="Drawdowns",
                yaxis_kwargs=dict(title="Value"),
                plot_func='plot_drawdowns',
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=['portfolio', 'value', 'drawdowns']
            ),
            underwater=dict(
                title="Underwater",
                yaxis_kwargs=dict(title="Drawdown"),
                plot_func='plot_underwater',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'value', 'drawdowns']
            ),
            gross_exposure=dict(
                title="Gross Exposure",
                yaxis_kwargs=dict(title="Gross exposure"),
                plot_func='plot_gross_exposure',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'exposure']
            ),
            net_exposure=dict(
                title="Net Exposure",
                yaxis_kwargs=dict(title="Net exposure"),
                plot_func='plot_net_exposure',
                pass_add_trace_kwargs=True,
                tags=['portfolio', 'exposure']
            )
        )
    )

    plot = PlotsBuilderMixin.plots

    @property
    def subplots(self) -> Config:
        return self._subplots


Portfolio.override_metrics_doc(__pdoc__)
Portfolio.override_subplots_doc(__pdoc__)

__pdoc__['Portfolio.plot'] = "See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."
