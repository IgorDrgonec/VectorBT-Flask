---
title: From signals
description: Documentation on portfolio simulation based on signals
icon: material/broadcast
---

# :material-broadcast: From signals

The method [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) (FO),
which was discussed previously, is the most primitive simulation method: it takes order information
in form of multiple array-like arguments and broadcasts them to a single shape, such that we
know exactly what has to be ordered of each asset at each bar. This method requires us
to have that information in advance, regardless of any events during the simulation. But what
if we wanted to create an order only given that we're not currently in the market, or in general,
to make an order dependable on the current simulation state? Such a conditional logic cannot be represented 
using orders alone - we either need to use a callback, or define more arrays. The former and the 
latter are both implemented by [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) (FS).

Before we dive deep into this method, make sure to learn more about signals [here](/tutorials/signal-development).
In a nutshell: signals are an abstraction layer over orders. Each signal consists of four boolean values:
:one: long entry, :two: long exit, :three: short entry, and :four: short exit. A combination of these 
values enables us to control the direction of an order relative to the current position. For example,
a short entry flag will reverse the current long position, or open a new short one if we're not in the market.
This way, position management can be abstracted away from order management, such that we can lean
back and only express our decision of whether we're currently bullish or bearish - a perfect playground
for ML models, by the way.

And there's another reason to love signals: statistically, in an entire universe of signal 
[permutations](https://en.wikipedia.org/wiki/Permutation), there is at least one permutation that beats 
the market all the time. This means that we can design a perfect trading algorithm using the above signal 
schema alone - we just need to guess the right timing and direction of each signal, which reduces
the number of factors we need to care about to just two (in a perfect world, of course, because in the
real world we also need to take into account the risk, execution modalities, etc.). For example, if the price 
of any security is $21 at day 1, $20 at day 2, and $22 at day 3, we can short entry at day 1 and long entry 
at day 2 to get positive-only returns. That's why trading systems and their backtesting components as well 
shouldn't necessarily scream from complexity in order to be profitable - they just need a robust signal 
generator as their algorithmic backbone and a trading infrastructure that closely matches the backtesting one.

!!! example
    Here are all the signal permutations for a price series with 4 points and their total return:

    ```pycon
    >>> import numpy as np
    >>> from itertools import product
    
    >>> price = np.array([21, 20, 22, 21])
    >>> returns = (price[1:] - price[:-1]) / price[:-1]
    >>> permutations = list(product([False, True], repeat=len(returns)))
    >>> total_return = np.prod(1 + np.where(permutations, returns, -returns), axis=1) - 1
    >>> pd.Series(total_return, index=permutations).sort_values(ascending=False)
    (False, True, False)     0.204762
    (False, True, True)      0.100000
    (True, True, False)      0.095238
    (True, True, True)       0.000000
    (False, False, False)   -0.014286
    (False, False, True)    -0.100000
    (True, False, False)    -0.103896
    (True, False, True)     -0.181818
    dtype: float64
    ```

    Don't run this on longer price series since the number of permutations grows exponentially 
    with the number of data points - `2^n`. That is, a year of daily history would require checking 
    `2^365` or `7.515336e+109` permutations.

## Mechanics

Similarly to FO, this method is also a class method of [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)
and has two Numba-compiled core functions: [simulate_from_signals_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.simulate_from_signals_nb)
and [simulate_from_signal_func_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb).
In fact, FS shares many arguments with FO, especially those used to set up the simulation, such as `init_cash`, 
and those carrying order information, such as `size`. For instance, if we look at the API documentation 
of the argument `size` under [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals), 
we'll see *"See `Portfolio.from_orders`"*. But also the simulation procedure of FS itself is very similar
to that of FO: while looping over all columns and rows, at each iteration, it resolves the current order 
and executes it by appending information on the filled order to the order records and updating the current 
simulation state. But that's where the similarities end.

### Framework

Here's an abstract visualization of the framework of FS run on three rows and 
two groups with two columns and one column respectively:

![](/assets/images/from_signals_framework.svg)

If you worked with vectorbt long enough, you have likely noticed that the framework of
[Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
follows that of [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
and [Portfolio.from_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_order_func).
Like most things in the vectorbt's universe, the simulation with FS is done by iterating over a 
so-called "target shape". This shape consists of two dimensions: rows representing the time axis and columns 
representing the asset axis (or, generally, configurations). Columns are further divided into groups: if multiple 
columns share the same cash, they are put into the same group (blue rectangle on the left above), while 
columns without cash sharing or grouping are isolated and appear as a group with exactly one column 
(blue rectangle on the right above). Groups are considered to be separate, atomic backtesting instances 
that aren't connected by any means, that is, splitting the shape by target groups shouldn't affect the 
final result. This, by the way, is why chunking is generally performed on groups rather than columns :bulb:

The actual iteration over the rows and groups happens in the 
[column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order):
the simulator starts moving over the rows in the first group, and once finished, continues with
the second group. Every time it hits a new row when processing a group, all the assets at this row
are called a "segment" because they together compete for the same resources at the same time,
or are connected by any user-defined means. For example, an account with `BTC-USD` and `ETH-USD` on 
the date `2020-01-01` is considered a segment because the value of both assets adds to the total value 
of the group at this date. Each asset within a segment is called an "element", which is the smallest 
simulation unit. An element in FS can host only one order, such that the number of filled orders is 
effectively capped by the number of rows times the number of columns. For example, a year of the daily 
`BTC-USD` and `ETH-USD` history can generate at most `365 * 2 = 730` orders, or one order per bar and asset.

#### Segment workflow

Segment is where the major part of the simulation takes place:

```mermaid
flowchart TD;
    id0["Update state at open"]
    
    id4["Get signals (0)"]
    id5["Get signals (1)"]
    id6["Get signals (2)"]
    
    id7["Generate order"]
    id8["Generate order"]
    id9["Generate order"]
    
    id10["Sort orders"]
    
    id11["Execute order (2)"]
    id12["Execute order (0)"]
    id13["Execute order (1)"]
    
    id14["Update state at close"]
    
    id15["Post-segment processing (optional)"]
    
    id0 --> id4;
    id0 --> id5;
    id0 --> id6;
    id4 --> id7;
    id5 --> id8;
    id6 --> id9;
    id7 --> id10;
    id8 --> id10;
    id9 --> id10;
    id10 --> id11;
    id10 --> id12;
    id10 --> id13;
    id11 --> id14;
    id12 --> id14;
    id13 --> id14;
    id14 --> id15;
```

FS first updates the current simulation state using the opening price. This is done to get
the group value in case some order has the size defined as a (target) percentage, to be able
to convert the size into an absolute amount of units. Then, it iterates through the columns in 
the current group and determines the four signals under each column. Those signals then get
converted into an order specification that is quite similar to the one FO takes. 

After having resolved all the order specifications, and if the automatic call sequence is enabled, the 
simulator attempts to sort the orders by their potential value such that sell orders are executed first - 
which is needed for rebalancing. This is only possible if all the (limit, order, and user-defined) orders within 
the current group happen either at the beginning or at the end of the current bar. If some orders happen in 
the middle of the bar, or some happen at the beginning and some at the end of a bar, an error will be 
thrown because we can only sort orders if they are guaranteed to happen at the same time! If the dynamic
call sequence is disabled and there are orders in multiple bar zones, the simulator will sort them 
by the bar zone. 

Finally, FS goes through the columns in a newly established call sequence to execute the orders, 
and updates the simulation state once again using the closing price. The final update is required
to optionally generate the portfolio returns and for the segment post-processing function.

### Signal generation

FS supports two signal generation modes: fixed (cached), and dynamic (non-cached). The first
mode is implemented by the function [simulate_from_signals_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.simulate_from_signals_nb),
which takes the signals as four pre-defined arrays and doesn't allow defining any callbacks, thus 
it's perfectly cacheable and doesn't have to be recompiled with each new runtime, unless it discovers
a new set of data types. The second mode is implemented by the function 
[simulate_from_signal_func_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.simulate_from_signal_func_nb),
which doesn't accept any signal arrays, but defines a signal function - a special callback meant to generate 
the four signals for each asset and at each bar dynamically. It's most suited for use cases where
the signal depends on the current simulation state. Furthermore, it defines a callback that is getting
called after processing the current segment, which can be used to pre-compute various metrics,
such as the Sharpe ratio. The main downsize is that it cannot be cached (yet), thus it has
to be re-compiled in each new runtime (:coffin: to those running vectorbt as a script).

The convenience of the method [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals),
which wraps those two modes, is in its ability to choose the mode automatically: whenever we override
any default callback, it runs the second mode over the first one.

#### Signal function

Remember how in FO we had to provide everything as arrays and could neither dynamically change the 
provided information nor affect the execution in any way? FS is much more flexible than that: it expects 
most information to be defined beforehand (acting as a facade), while signals can be generated both
statically and dynamically. Let's play with the dynamic signal generation a bit.

The second mode is implemented by accepting a user-defined callback function, `signal_func_nb`. 
Whenever the main simulation loop hits a new row (bar), it asks each asset in the current group to 
generate signals using this callback function. For this, it packs all the information that might be 
useful to the user, such as the current cash balance and the group value, into a named tuple of the 
type [SignalContext](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext). In return, 
it expects the function to return four signals, which will be used to create an order for that asset.

Here's how a dead-simple signal function that orders nothing looks like:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> import vectorbtpro as vbt

>>> @njit
... def signal_func_nb(c):
...     return False, False, False, False

>>> close = pd.DataFrame({
...     "BTC-USD": [20594.29, 20719.41, 19986.60, 21084.64], 
...     "ETH-USD": [1127.51, 1125.37, 1051.32, 1143.20],
...     "DOT-USD": [7.88, 7.74, 7.41, 7.78], 
...     "BNB-USD": [216.90, 219.67, 214.23, 228.92]
... })

>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     signal_func_nb=signal_func_nb
... )
>>> pf.order_records
array([], dtype={...})
```

!!! hint
    To avoid waiting for the function to compile, remove the `@njit` decorator from `signal_func_nb` 
    and pass `jitted=False` to `from_signals` in order to disable Numba for this method completely.
    Do this only if the amount of input data is small (< 1000).

To better understand when the function is called, let's expand our data to two assets and print out 
the current column and row:

```pycon
>>> @njit
... def signal_func_nb(c):
...     print(c.col, c.i)
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     close=close[["BTC-USD", "ETH-USD"]], 
...     signal_func_nb=signal_func_nb
... )
0 0
0 1
0 2
0 3
1 0
1 1
1 2
1 3
```

We see that the function was called at each row, first in the column `BTC-USD`, then in the column `ETH-USD`.
Here, both assets are acting as isolated tests, thus the simulator processes one column after another.
But once we introduce a grouping with or without cash sharing, which binds columns semantically, 
the simulator will process the columns group-wise, that is, it will move over the groups, then over the 
rows, and finally over the columns in the current group and at the current row. Let's demonstrate this
by introducing two groups with two assets sharing the same cash:

```pycon
>>> @njit
... def signal_func_nb(c):
...     print(c.group, c.col, c.i)
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     signal_func_nb=signal_func_nb,
...     group_by=[0, 0, 1, 1],
...     cash_sharing=True
... )
0 0 0
0 1 0
0 0 1
0 1 1
0 0 2
0 1 2
0 0 3
0 1 3
1 2 0
1 3 0
1 2 1
1 3 1
1 2 2
1 3 2
1 2 3
1 3 3
```

The context tuple passed to the signal function contains all the necessary information to
identify the position of the call in the simulation. For example, we can use `c.index[c.i]` with
[SignalContext.index](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.index)
and [SignalContext.i](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.i)
to get the timestamp of the current bar. We can also change the current state of any pending limit
or stop order before it's processed since the signal function is conceptually executed
right before the beginning of the bar.

Thanks to the strict processing of groups from left to right and storing the state of each group globally, 
we can access the order records and, generally, the latest simulation state of all the groups 
that were processed earlier. For example, [SignalContext.last_cash](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.last_cash)
has the same number of elements as there are groups. This is powerful and dangerous at the same time:
we can introduce complex intergroup relationships if wanted, or accidentally access the wrong group 
if not paying enough attention.

### Signal resolution

Signals are just one additional level of abstraction over orders, meaning there needs to be some logic 
in place that translates them into order specifications. Indeed, whenever the simulator receives a new 
set of four signals at each row and column, it first resolves them into a single signal, which then 
gets converted into an order. The resolution step checks whether the provided signals have conflicts. 
Mostly, the signals are expected to have only one `True` value and three `False` values, but sometimes
multiple signals are `True`, especially when the signal function is forwarding data from multiple boolean arrays.
In such a case, the simulator goes through the following procedure consisting of multiple checks and fixes:

```mermaid
flowchart TD;
    id1["Long entry and long exit?"]
    id2["Short entry and short exit?"]
    id3["Long entry and short entry?"]
    id4["Long/short position and short/long entry?"]
    id5["Final signal"]
    
    id1 -->|"fix"| id2;
    id2 -->|"fix"| id3;
    id3 -->|"fix"| id4;
    id4 -->|"fix"| id5;
```

It first checks whether there are multiple `True` values within the same direction,
for example, when the long entry and long exit are both set. To decide which one in the long direction to keep,
it looks at the argument `upon_long_conflict` of the type [ConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.ConflictMode)
provided by the user. For example, the option "adjacent" will pick the signal adjacent to 
the position we are currently in such that only the long entry will remain active if we're in a long position.
This is done by calling the function [resolve_signal_conflict_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_signal_conflict_nb):

```pycon
>>> vbt.pf_nb.resolve_signal_conflict_nb(
...     position_now=20,
...     is_entry=True,
...     is_exit=True,
...     direction=vbt.pf_enums.Direction.LongOnly,
...     conflict_mode=vbt.pf_enums.ConflictMode.Adjacent
... )
(True, False)
```

After deciding for at most one signal in both directions, the simulator checks whether both the long 
and short entry are active and uses the function [resolve_dir_conflict_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_dir_conflict_nb)
based on the argument `upon_dir_conflict` of the type [DirectionConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.DirectionConflictMode) 
to select the winner. For example, we can choose to always go short when there is any uncertainty:

```pycon
>>> vbt.pf_nb.resolve_dir_conflict_nb(
...     position_now=20,
...     is_long_entry=True,
...     is_short_entry=True,
...     upon_dir_conflict=vbt.pf_enums.DirectionConflictMode.Short,
... )
(False, True)
```

Finally, it calls the function [resolve_opposite_entry_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_opposite_entry_nb)
if there is an entry signal that is opposite to the direction of the current position. For example,
if we're in a long position and the short entry signal is set, the simulator will use the argument
`upon_opposite_entry` of the type [OppositeEntryMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OppositeEntryMode) 
to decide whether to reduce, close, or completely reverse the current long position.
Let's make the short entry signal behave like the long exit signal:

```pycon
>>> vbt.pf_nb.resolve_opposite_entry_nb(
...     position_now=20,
...     is_long_entry=False,
...     is_long_exit=False,
...     is_short_entry=True,
...     is_short_exit=False,
...     upon_opposite_entry=vbt.pf_enums.OppositeEntryMode.Close,
...     accumulate=False,  # (1)!
... )
(False, True, False, False, 0)
```

1. This argument can be `True`, `False`, or any of [AccumulationMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccumulationMode) 

In the end, only one active signal out of four will remain :railway_track:

### Signal conversion

We have our one signal, now what? It's time to convert it into an order! And this is the easiest step
in the pipeline, done by the function [signal_to_size_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.signal_to_size_nb),
which takes the four signals (three of which are now deactivated) and the size requirement under this row
and column (i.e., for this element), and returns the order size, size type, and direction to be requested.
For example, being in a position of 20 shares and receiving the long exit signal, the 
size becomes minus 20 shares, the size type becomes [SizeType.Amount](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SizeType.Amount),
and the direction becomes [Direction.LongOnly](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Direction.LongOnly):

```pycon
>>> vbt.pf_nb.signal_to_size_nb(
...     position_now=20,
...     val_price_now=20594.29,  # (1)!
...     value_now=411885.80,  # (2)!
...     is_long_entry=False,
...     is_long_exit=True,
...     is_short_entry=False,
...     is_short_exit=False,
...     size=0.1,  # (3)!
...     size_type=vbt.pf_enums.SizeType.ValuePercent,  # (4)!
...     accumulate=False  # (5)!
... )
(-20.0, 0, 0)
```

1. The latest asset price known
2. The latest group value known
3. Default value for this element
4. Default value for this element
5. Default value for this element

Even though we provided the function with the default order specification for the current element, 
such as `size`, the function didn't use it because it isn't required for closing the current position.
On the other hand, if we wanted to reverse the current position (that is, close it and then order
using the default specification), those inputs would suddenly become effective:

```pycon
>>> vbt.pf_nb.signal_to_size_nb(
...     position_now=20,
...     val_price_now=20594.29,
...     value_now=411885.80,
...     is_long_entry=False,
...     is_long_exit=False,
...     is_short_entry=True,  # (1)!
...     is_short_exit=False,
...     size=0.1,
...     size_type=vbt.pf_enums.SizeType.ValuePercent,
...     accumulate=False
... )
(-22.0, 0, 2)
```

1. Reverse the long position

The size is calculated as follows: reduce the number of shares by 20 to close out the long position,
and, given that we're operating with a percentage of the current group value, open a new short position 
of `size * value_now / val_price_now = 2.0` shares. The size type is [SizeType.Amount](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SizeType.Amount)
while the direction is [Direction.Both](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Direction.Both)
because the operation now involves two directions.

### Main order resolution

The simulator called the signal function, resolved the incoming signals, and converted them into
an order specification. But this is not the only order that competes for the current bar: there may be 
also pending limit and stop orders. Since the FS simulation function can process at most one order at 
each bar, it has to decide for a winner, which should always be an order that executes first.
But how do we know which one comes first if we don't have any intra-bar data? We can
still divide each bar into three "zones": opening (first rectangle below), somewhere in-between
(second rectangle below), and closing (third rectangle below). For example, if a stop order was hit at or 
before the opening of the current bar and the user order should execute at the close price, then clearly the 
stop order should be first on the line. Here's the full decision chain:

```mermaid
flowchart TD;
    subgraph " "
    id1["Pending limit price hit at/before open?"]
    id2["Pending stop price hit at/before open?"]
    id3["User price is open?"]
    
    id1 -->|"no"| id2;
    id2 -->|"no"| id3;
    end
    
    subgraph " "
    id4["Pending limit price hit?"]
    id5["Pending stop price hit before close?"]
    id6["User price isn't close?"]
    
    id3 -->|"no"| id4;
    id4 -->|"no"| id5;
    id5 -->|"no"| id6;
    end
    
    subgraph " "
    id7["Pending stop price hit at close?"]
    id8["User price is close?"]
    
    id6 -->|"no"| id7;
    id7 -->|"no"| id8;
    end
```

As we see, limit orders have priority over stop orders, while stop orders have priority 
over user orders, but only if they are triggered within the same zone of a bar.

### Limit management

While a market order is a transaction meant to execute as quickly as possible at the current market price, 
a limit order is an order to buy or sell an asset with a restriction on the maximum price to be paid or 
the minimum price to be received (the "limit price"). The price of a limit order is getting compared 
against a pre-defined price level. Until this level is reached, the order is marked as pending, and 
it will not be filled if the price does not reach this level.

#### Creation

Whenever a stop or user-defined order goes through and its order type provided via `order_type` 
is [OrderType.Limit](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderType.Limit), 
the simulator first determines what kind of limit price the order should be executed at: open, close, or 
something else? And this is a very important concept to grasp: the argument `price` gives
vectorbt hints on where in the bar the operation should take place. If the limit price is the open price
(provided as either [PriceType.Open](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceType.Open)
or `-np.inf`), the simulator is allowed to use the entire candle for its checks and execute the order right
upon detecting the price being hit, at the same bar. If the limit price is not the close price but 
something in-between, the simulator is allowed to use the close price only. If the limit 
price is the close price (provided as either [PriceType.Close](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceType.Close) or `np.inf`),
the simulator is not allowed to execute the limit order right away - it's forced to postpone its 
very first check to the next bar.

If the limit order couldn't be executed at the same bar as it was created, the order is marked as 
pending and all the relevant information becomes stored in a record array of the type 
[limit_info_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.limit_info_dt),
which is structured by asset. This array can keep only one instance per asset, thus FS allows
only one limit order to be active at a time. In a signal function, this array can be accessed
via `c.limit_info_dt`, allowing us to change any information before the new bar. For example,
to change the price: `c.limit_info_dt["init_price"][c.col] = new_price`.

#### Expiration

Once the simulator hits the next bar, it first uses [check_limit_expired_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.check_limit_expired_nb)
to check whether the pending limit order expired at the beginning or somewhere in the middle of the bar.
If the former, the order is thrown away. If the latter, the simulator also checks whether the order 
was hit at the beginning of the bar (and execute it), and if not, throw it away because there is no
guarantee that the order was hit before the deadline. For example, let's assume that the order can be 
at most 36 hours in force, it was issued at the day `2020-01-01`, and now is the day `2020-01-02`:

```pycon
>>> vbt.pf_nb.check_limit_expired_nb(
...     creation_idx=0,
...     i=1,
...     tif=pd.Timedelta("36h").to_timedelta64().astype(np.int64),  # (1)!
...     index=pd.date_range("2020-01-01", periods=3).values.astype(np.int64),
...     freq=pd.Timedelta("1d").to_timedelta64().astype(np.int64)
... )
(False, True)
```

1. Index and time deltas must be converted into nanoseconds

We see that the function marked the order as expired, but not at the beginning of the bar
such that it can still be executed using the open price. But if the lifespan of the order was 24 hours, 
the function would also raise the first flag and disallow any execution:

```pycon
>>> vbt.pf_nb.check_limit_expired_nb(
...     creation_idx=0,
...     i=1,
...     tif=pd.Timedelta("24h").to_timedelta64().astype(np.int64),
...     index=pd.date_range("2020-01-01", periods=3).values.astype(np.int64),
...     freq=pd.Timedelta("1d").to_timedelta64().astype(np.int64)
... )
(True, True)
```

!!! info
    The lifespan is calculated by subtracting any time from the opening time of the creation bar,
    even if the order was placed at the very end of the creation bar.

#### Activation

Once we're sure that the order **can** be executed at this bar (i.e., it won't expire), the simulator uses 
the function [check_limit_hit_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.check_limit_hit_nb)
to check whether the order **should** be executed by determining whether its target price has been hit. 
This is implemented through comparison of the price against the current candle. For example, if we have a 
pending buy limit order with a target price of `9.5`, the function will check whether the low price went 
below the target price:

```pycon
>>> vbt.pf_nb.check_limit_hit_nb(
...     open=10.0,
...     high=11.0,
...     low=9.0,
...     close=10.5,
...     price=9.5,
...     size=2.0
... )
(9.5, False, True)
```

If the target price was `11`, the function would notify us about the price being hit already 
at the beginning of the bar; in such a case, the order will be executed right away using the open price:

```pycon
>>> vbt.pf_nb.check_limit_hit_nb(
...     open=10.0,
...     high=11.0,
...     low=9.0,
...     close=10.5,
...     price=11.0,
...     size=2.0
... )
(10.0, True, True)
```

#### Cancellation

If the target price hasn't been hit, the limit order remains pending. It can still be cancelled
manually in the signal function called before all the checks above, or in the post-segment function
called after processing the entire segment. The pending order will also be cancelled automatically 
once any stop order gets executed since the latter may change the simulation state and potentially pull 
the resources required to execute the former in the future. 

Finally, the four signals returned by the signal function and resolved into a single signal also can affect 
the pending order, regardless of whether the final signal gets executed or not. Consider the example where 
we have a pending buy limit order and the user decides to issue the long exit or short entry signal; in this case, 
the most intuitive reaction would be cancelling the pending order since the user have changed their mind. 
Exactly this is happening by default. Such "pending conflicts" are resolved using the function 
[resolve_pending_conflict_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_pending_conflict_nb),
which uses the arguments `upon_adj_limit_conflict` and `upon_opp_limit_conflict`,
both of the type [PendingConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PendingConflictMode),
to decide what to do if the direction of the pending order is adjacent and opposite respectively 
to the direction of the resolved user-defined signal.

```pycon
>>> vbt.pf_nb.resolve_pending_conflict_nb(
...     is_pending_long=True,
...     is_user_long=False,
...     upon_adj_conflict=vbt.pf_enums.PendingConflictMode.KeepIgnore,
...     upon_opp_conflict=vbt.pf_enums.PendingConflictMode.CancelIgnore,
... )
(False, False)
```

Here, the function decided to cancel the limit order and to ignore the user-defined signal.

### Stop management

Stop orders help to ensure a higher probability of achieving a predetermined entry or exit price, 
limiting the investor's loss, or locking in a profit. They remain dormant until a certain price is passed, 
at which time they are activated as a market or limit order. Execution of a stop order usually closes 
out the position. 

#### Types

There are four types of stop orders:

1. Stop loss (SL)
2. Trailing stop loss (TSL)
3. Trailing take profit (TTP)
4. Take profit (TP)

By using a stop-loss order, we're limiting our risk in the trade to a set amount in the event that 
the market moves against us. For instance, if a stop-loss sell order were placed at $45 per unit, 
the order would be inactive until the price reached or dropped below $45. The order would then be 
transformed into a market or limit order, and the units would be sold at the best available price.
A take profit is pretty much the exact opposite. It expresses how much we're willing to make as a 
profit with one trade and close it once we're happy with the amount. A combination of an SL 
and TP order creates a certain risk-to-reward ratio, which can be further tuned to respect the odds of 
reaching each certain breakout scenario. 

A different bread are trailing orders: when the price increases, it drags the trailing stop along with it. 
Then, when the price finally stops rising, the new stop-loss price remains at the level it was dragged to, 
thus automatically protecting our downside, while locking in profits as the price reaches new highs.
Also, TTP is a version of TSL that gets activated after a certain threshold. These two orders are
usually viewed and represented as a single order.

#### Creation

In contrast to limit orders, stop orders are created after an entry order has been filled,
and behave identically to user-defined exit signals that are triggered once a stop condition has been met.
An entry order is any successfully-filled order that has opened a new position or increased an existing one.
Once the simulator detects such an order, it first resolves the stop entry price provided via the
argument `stop_entry_price` of the type [StopEntryPrice](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopEntryPrice).
This is the initial price to which all the stop values and thresholds are applied.

!!! note
    By default, the stop entry price is the closing price, not the order price, to avoid
    a scenario where the stop is getting hit at the very first bar but cannot be executed
    since we don't have intra-bar data and cannot execute two orders at the same bar.
    By using the order price, the earliest time the stop can execute at is the opening of the next bar.

Based on this price, the simulator can also determine where exactly in the bar the stop order should be triggered. 
Why is this important? Because the simulator needs to know whether it can already use the current candle 
to update the price of any TSL and TTP order. Internally, the information on stop orders is stored inside 
three arrays of the data types [sl_info_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.sl_info_dt), 
[tsl_info_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.tsl_info_dt), and 
[tp_info_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.tp_info_dt), laid out per asset.
Each data type has a similar schema: the initial row and price, the current stop value (either
in absolute or percentage terms), and the limit delta and its format if the stop order should ultimately 
trigger a limit order. The trailing stops also include the newly updated price and the row of the update.

#### Activation

Stop orders cannot be activated at the same bar as they were issued, even if the entry price 
is the opening price. This is because FS cannot handle two orders at the same bar (if there is a need
for this, use a flexible order function with [Portfolio.from_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_order_func)).
After hitting a new bar, the price of any pending SL and TP order then gets checked against the low
and high price of the current candle respectively (the other way round for short positions),
using the function [check_stop_hit_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.check_stop_hit_nb),
which returns the hit price, whether it happened on open, and whether it happened at all.
For example, let's imagine that the stop entry price is $10 per unit and the stop loss sits at 10%.
For the stop to be marked as hit, the lowest price of this candle must be under `10 * (1 - 0.1) = 9` per unit:

```pycon
>>> vbt.pf_nb.check_stop_hit_nb(
...     open=10.0,
...     high=11.0,
...     low=9.0,
...     close=10.5,
...     is_position_long=True,
...     stop_price=10.0,
...     stop=0.1
... )
(9.0, False, True)
```

But if the entry price was at $12 per unit, the stop would match already at open:

```pycon
>>> vbt.pf_nb.check_stop_hit_nb(
...     open=10.0,
...     high=11.0,
...     low=9.0,
...     close=10.5,
...     is_position_long=True,
...     stop_price=12.0,
...     stop=0.1
... )
(10.0, True, True)
```

To check a TP order, we would need to set the argument `hit_below` to `False`.

As opposed to fixed stop orders, TSL and TTP orders additionally need to track the peak price
the stop price is based upon. Since we have no information on whether the highest price of a candle 
comes before the lowest price or vice versa, we need to split the candle into distinct zones and 
update the peak price in a zone that precedes that of the stop check. First, the simulator uses the 
opening price to update the peak price. Then, it checks whether the stop has been hit during the 
entire bar. If not, it proceeds to update the peak price using the highest (for long positions) 
or lowest (for short positions) price of the candle, and performs the stop check again, but now 
using the closing price alone to avoid the ambiguity that's been mentioned before. This way, the 
simulator always pessimistically assumes that the worst event (where the stop is being hit) happens 
before the best event (where the peak price is being updated).

To make the matter even more complex, a TTP order additionally needs to check whether its threshold
has been crossed to be properly activated. The check is performed using the function
[check_tsl_th_hit_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.check_tsl_th_hit_nb):
it takes the initial and the peak price, and checks whether the difference between them is greater 
than or equal to the threshold. If so, the order gets converted into a regular TSL order. 
But if the difference is smaller, the simulator proceeds to update the peak price using the current candle,
and makes a second attempt to check for the threshold crossover. If the difference is finally greater,
it uses [check_stop_hit_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.check_stop_hit_nb), 
but it cannot use the current candle anymore since it's unknown whether the stop has been hit after or 
before the threshold has been crossed, thus only the closing price is used by disabling the argument 
`can_use_ohlc`. Here's a depiction of what's happening:

```mermaid
flowchart TD;
    subgraph " "
    id0["Update peak price using open"]
    id1["Has a threshold?"]
    id2["Threshold crossed at/before open?"]
    id3["Stop hit at/after open?"]
    
    id0 --> id1;
    id1 -->|"yes - TTP"| id2;
    id1 -->|"no - TSL"| id3;
    id2 -->|"yes"| id3;
    end
    
    subgraph " "
    id4["Update peak price using high/low"]
    id5["Has a threshold?"]
    id6["Threshold crossed before close?"]
    id7["Stop hit before close?"]
    
    id2 -->|"no"| id4;
    id3 -->|"no"| id4;
    id4 --> id5;
    id5 -->|"yes - TTP"| id6;
    id5 -->|"no - TSL"| id7;
    id6 -->|"yes"| id7;
    end
    
    id3 -->|"yes"| id9;
    id6 -->|"no"| id8;
    id7 -->|"yes"| id9;
    id7 -->|"no"| id8;
    
    id8["Keep pending"]
    id9["Execute"]
```

What happens if multiple stops have been hit? The simulator pessimistically assumes that SL comes first,
TSL and TTP come second, and TP comes last. The pending stop that comes before others gets executed while
the remaining pending stops get canceled. 

#### Resolution

After that, the winner gets converted into the four signals using the function 
[generate_stop_signal_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.generate_stop_signal_nb),
which is based on the current position and the default stop exit behavior defined using
the argument `stop_exit_type` of the type [StopExitType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopExitType).
For example, instead of closing out the position, we can make the function reverse the position
by using `StopExitType.Reverse`:

```pycon
>>> vbt.pf_nb.generate_stop_signal_nb(
...     position_now=20,
...     stop_exit_type=vbt.pf_enums.StopExitType.Reverse
... )
(False, False, True, False, 0)
```

As we can see, the short entry signal is `True` while other signals are `False`. The number after the signals 
is the selected accumulation mode of the type [AccumulationMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccumulationMode) 
for a case where we want to reduce the position instead of closing out or reversing it. What's left is 
the resolution of the stop exit price, done by the function 
[resolve_stop_exit_price_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_stop_exit_price_nb).
The logic is quite simple: if the argument `stop_exit_price` of the type 
[StopExitPrice](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopExitPrice) is `StopExitPrice.Close`,
use the closing price, otherwise use the stop price that has been hit. The argument can also
be provided as an actual price.

```pycon
>>> vbt.pf_nb.resolve_stop_exit_price_nb(
...     stop_price=9.0,
...     close=10.5,
...     stop_exit_price=vbt.pf_enums.StopExitPrice.Stop
... )
9.0

>>> vbt.pf_nb.resolve_stop_exit_price_nb(
...     stop_price=9.0,
...     close=10.5,
...     stop_exit_price=9.5
... )
9.5
```

Finally, the order signal gets converted into a market or limit order specification the same way 
as a user-defined signal. See [Signal conversion](#signal-conversion).

#### Updating

Apart from the possibility to update any stop in a callback, there is a possibility to update
the stop automatically once the current position has increased. In such a case, the 
argument `upon_stop_update` of the type [StopUpdateMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopUpdateMode) 
controls whether any current stop should remain or be reset. This decision is done by the function
[should_update_stop_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.should_update_stop_nb).
For example, our position has increased, and we want to know whether the current stop should
be updated given the option `StopUpdateMode.Override`:

```pycon
>>> vbt.pf_nb.should_update_stop_nb(
...     new_stop=0.1,
...     upon_stop_update=vbt.pf_enums.StopUpdateMode.Override
... )
True
```

But if the new stop value is NaN (i.e., no stop), we shouldn't update:

```pycon
>>> vbt.pf_nb.should_update_stop_nb(
...     new_stop=np.nan,
...     upon_stop_update=vbt.pf_enums.StopUpdateMode.Override
... )
False
```

Unless the option is `StopUpdateMode.OverrideNaN`, which will effectively disable all stops:

```pycon
>>> vbt.pf_nb.should_update_stop_nb(
...     new_stop=np.nan,
...     upon_stop_update=vbt.pf_enums.StopUpdateMode.OverrideNaN
... )
True
```

This won't apply to the case where the current position has decreased. But why should we care
about updates if signals usually either open or close positions, and don't increase or decrease them? 
When using accumulation, signals can add to the position or remove from the position.
In this case, we should ask ourselves: should this position change invalidate the stops we defined previously,
or define new ones? The argument `upon_stop_update` controls this behavior.

#### Cancellation

Similarly to updating, cancellation of the currently pending stop orders happens when the position
has been closed out. This will clear all the stops automatically. But also similarly to limit orders,
there is a possibility of a conflict with an active user-defined signal, which is resolved using the function 
[resolve_pending_conflict_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.resolve_pending_conflict_nb).
The arguments that are used to resolve any pending conflicts for stop orders are `upon_adj_stop_conflict` 
and `upon_opp_stop_conflict`, both of the type [PendingConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PendingConflictMode).
For example, if the user has decided to reduce the position and clear all the pending stops
along with it, they can set the argument `upon_adj_stop_conflict` (reducing the position and attempting
to close it are considered adjacent signals) to the option `PendingConflictMode.CancelExecute`.

## Signals

We've covered some theory on how this simulation method works. Let's take a break from reading
and concern ourselves with signal arrays, which together with a signal function are the main input
to this method. As we already know, signals come in two flavors:

1. Direction-unaware signals: entries and exits enhanced by direction
2. Direction-aware signals: Long entries, long exits, short entries, short exits

The first flavor is a compressed form of the second flavor, so to say: we can always 
convert direction-unaware signals into direction-aware, but not the other way round since
the first format defines a total of `2 * 2 * 3 = 12` combinations and the second format defines
a total of `2 * 2 * 2 * 2 = 16` combinations. On the other hand, the first format is easier to use 
since we can set the direction globally and just operate with two arrays instead of four.

But first, let's fetch the entire history of `BTCUSDT` and `ETHUSDT` for our examples below:

```pycon
>>> data = vbt.BinanceData.fetch(["BTCUSDT", "ETHUSDT"])
```

As we won't need the entire history to illustrate most concepts, let's select the week of data
between the 18th and 24th February 2021, where there is a substantial change in price in both directions:

```pycon
>>> sub_data = data.loc["2021-02-18":"2021-02-24"]

>>> vbt.settings.set_theme("dark")
>>> sub_data.plot(symbol="BTCUSDT")
```

![](/assets/images/from_signals_sub_data.svg)

Let's try passing the data without signals:

```pycon
>>> pf = vbt.Portfolio.from_signals(sub_data)  # (1)!
>>> pf.orders.count()
symbol
BTCUSDT    0
ETHUSDT    0
Name: count, dtype: int64
```

1. Even though the first argument expects the closing price (`close`),
we can pass the entire data instance, from which the OHLC features will be extracted automatically

By default, all signals are set to `False`, thus no orders were generated. 

Now, let's assume that our ML model has correctly predicted the peak on the 21st February and ordered 
us to enter a position on the 18th February and close it on the 21st February. For this, we need to 
build our entry and exit arrays of the same shape as our data. But instead of specifying the same 
signals for each asset redundantly, we can provide a Series instead of a DataFrame, which will be 
applied to each asset thanks to [broadcasting](/documentation/portfolio/from-orders/#broadcasting):

```pycon
>>> X = True  # (1)!
>>> O = False

>>> entries = pd.Series([X, O, O, O, O, O, O])
>>> exits   = pd.Series([O, O, O, X, O, O, O])
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=entries,  # (2)!
...     exits=exits
... )
>>> pf.orders.records_readable
   Order Id   Column              Signal Index            Creation Index  \\
0         0  BTCUSDT 2021-02-18 00:00:00+00:00 2021-02-18 00:00:00+00:00   
1         1  BTCUSDT 2021-02-21 00:00:00+00:00 2021-02-21 00:00:00+00:00   
2         0  ETHUSDT 2021-02-18 00:00:00+00:00 2021-02-18 00:00:00+00:00   
3         1  ETHUSDT 2021-02-21 00:00:00+00:00 2021-02-21 00:00:00+00:00   

                 Fill Index      Size     Price  Fees  Side    Type Stop Type  
0 2021-02-18 00:00:00+00:00  0.001940  51552.60   0.0   Buy  Market      None  
1 2021-02-21 00:00:00+00:00  0.001940  57408.57   0.0  Sell  Market      None  
2 2021-02-18 00:00:00+00:00  0.051557   1939.61   0.0   Buy  Market      None  
3 2021-02-21 00:00:00+00:00  0.051557   1933.53   0.0  Sell  Market      None  
```

1. Shortcuts for better readability
2. We could have also specified it as a two-dimensional NumPy array with one column

We can see that the first order in the column `BTCUSDT` is a buy market order that opened
a new long position. The second order is of the same size but the opposite side, thus it was
used to close out the long position. Reading orders isn't always straight-forward, especially
when it comes to determining when positions are opened and closed. To get a better overview,
let's calculate and print the position for each symbol at each bar:

```pycon
>>> pf.assets
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00194  0.051557
2021-02-20 00:00:00+00:00  0.00194  0.051557
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

The returned array represents the position at the end of each bar, hence we're still in the 
market on the 20th February but went out of the market on the 21st February.

We provided the same array for each symbol, but what if our ML model told us that the peak
for `ETHUSDT` is one day ahead of that for `BTCUSDT`? As soon as our signal specification starts varying
with columns, we need to build the respective signal array as a DataFrame with values defined per element. 
Let's keep the entry array the same for both symbols (entry signals do not vary with columns in our case) 
and only expand the exit array:

```pycon
>>> entries = pd.Series([X, O, O, O, O, O, O])
>>> exits = pd.DataFrame({
...     0: [O, O, O, X, O, O, O],
...     1: [O, O, X, O, O, O, O],
... })  # (1)!
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=entries,
...     exits=exits
... )
>>> pf.assets
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00194  0.051557
2021-02-20 00:00:00+00:00  0.00194  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

1. We could have also used symbols as column names. Defining rows and columns as a simple
range of values (from `0` to `n`) will ignore the labels and broadcast only using shapes.

We can now observe that the long position in the column `ETHUSDT` was cleared one day before
that of `BTCUSDT`, as our imaginary model wished. To make array creation a bit easier and 
to avoid setting each element manually, we can use the symbol wrapper of the data instance
to create empty boolean arrays of the same shape as the data, and fill them on specific dates:

```pycon
>>> exits = sub_data.symbol_wrapper.fill(False)  # (1)!
>>> exits.loc["2021-02-21", "BTCUSDT"] = True
>>> exits.loc["2021-02-20", "ETHUSDT"] = True
>>> exits
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00    False    False
2021-02-19 00:00:00+00:00    False    False
2021-02-20 00:00:00+00:00    False     True
2021-02-21 00:00:00+00:00     True    False
2021-02-22 00:00:00+00:00    False    False
2021-02-23 00:00:00+00:00    False    False
2021-02-24 00:00:00+00:00    False    False
```

1. Create an array with the same number of columns as we have symbols, and fill it with `False`

For readers who appreciate hot features, here's how to let the vectorbt's broadcaster create
both arrays and fill them dynamically using [index dictionaries](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.fill_using_index_dict):

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=vbt.index_dict({0: True}),
...     exits=vbt.index_dict({
...         vbt.ElemIdx("2021-02-21", "BTCUSDT"): True,
...         vbt.ElemIdx("2021-02-20", "ETHUSDT"): True
...     })
... )
>>> pf.assets
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00194  0.051557
2021-02-20 00:00:00+00:00  0.00194  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

The best in the approach above is that the broadcaster won't create arrays bigger than necessary:
it will notice that the entry specification is the same for both symbols and create
an array with one column instead of two, saving us some memory.

### Direction-unaware

In all the examples above, we provided only two arrays: `entries` and `exits`. Whenever this happens,
the method treats the provided signals as direction-unaware, meaning that an additional
argument `direction` is used to control their direction. By default, the direction is `Direction.LongOnly`
(see `signal_direction` in [portfolio settings](/api/_settings/#vectorbtpro._settings.portfolio)).
To change the direction, override `direction` with any option available in 
[Direction](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Direction). An option can be provided
either as an integer or a field name. For example, let's allow *both* directions to 
reverse the position on exit instead of just closing it out:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=entries,
...     exits=exits,
...     direction="both"  # (1)!
... )
>>> pf.assets
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00194  0.051557
2021-02-20 00:00:00+00:00  0.00194 -0.051557
2021-02-21 00:00:00+00:00 -0.00194 -0.051557
2021-02-22 00:00:00+00:00 -0.00194 -0.051557
2021-02-23 00:00:00+00:00 -0.00194 -0.051557
2021-02-24 00:00:00+00:00 -0.00194 -0.051557
```

1. Can also be provided in the numeric format as `Direction.Both`

Zero values have changed to negative values, which means that positions are now being reversed. 
Similarly to the arguments for signals, the argument for direction can also be provided as an array. 
This makes possible defining different directions for different dates and symbols. For example, 
let's long `BTCUSDT` and short `ETHUSDT`. Thanks to broadcasting, to provide any information per column, 
we can represent it as a two-dimensional array with just one row:

```pycon
>>> direction = pd.DataFrame([["longonly", "shortonly"]])
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=entries,
...     exits=exits,
...     direction=direction
... )
>>> pf.assets
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194 -0.051557
2021-02-19 00:00:00+00:00  0.00194 -0.051557
2021-02-20 00:00:00+00:00  0.00194  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

The position values under the column `ETHUSDT` have turned negative indicating that we're 
in a short position. The following example illustrates the usage of direction defined per element 
by entering a long position at the first bar, exiting it at the peak, then entering a short one at 
the next bar, and finally exiting it at the last bar:

```pycon
>>> L = vbt.pf_enums.Direction.LongOnly  # (1)!
>>> S = vbt.pf_enums.Direction.ShortOnly

>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),  # (2)!
...     entries=  pd.Series([X, O, O, O, X, O, O]),
...     exits=    pd.Series([O, O, O, X, O, O, X]),
...     direction=pd.Series([L, L, L, L, S, S, S])
... )
>>> pf.assets
Open time
2021-02-18 00:00:00+00:00    0.001940
2021-02-19 00:00:00+00:00    0.001940
2021-02-20 00:00:00+00:00    0.001940
2021-02-21 00:00:00+00:00    0.000000
2021-02-22 00:00:00+00:00   -0.002059
2021-02-23 00:00:00+00:00   -0.002059
2021-02-24 00:00:00+00:00    0.000000
Freq: D, dtype: float64
```

1. Shortcuts for better readability, numeric format
2. Select one symbol of data

!!! note
    The numeric format should be preferred over the string format for bigger arrays since strings 
    must be converted to integers prior to the simulation, which is a relatively slow operation.

### Direction-aware

Direction-aware signals are a more flexible form of signals as they allow for more signal combinations.
To switch to this mode, we need to provide the arguments `short_entries` and `short_exits` acting as 
short signals along with the arguments `entries` and `exits` acting as long signals. This will disable 
the `direction` argument completely as the direction is now expressed by the signals themselves.
Let's adapt the example above:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     entries=      pd.Series([X, O, O, O, O, O, O]),
...     exits=        pd.Series([O, O, O, X, O, O, O]),
...     short_entries=pd.Series([O, O, O, O, X, O, O]),
...     short_exits=  pd.Series([O, O, O, O, O, O, X]),
... )
>>> pf.assets
Open time
2021-02-18 00:00:00+00:00    0.001940
2021-02-19 00:00:00+00:00    0.001940
2021-02-20 00:00:00+00:00    0.001940
2021-02-21 00:00:00+00:00    0.000000
2021-02-22 00:00:00+00:00   -0.002059
2021-02-23 00:00:00+00:00   -0.002059
2021-02-24 00:00:00+00:00    0.000000
Freq: D, dtype: float64
```

So, when do we use which signal mode? Use direction-unaware signals when working with a single
direction throughout the entire column, and direction-aware signals for more granular decisions,
especially when positions need to be closed when both directions are allowed, for example,
to close out any position at the end of a day.

### Signal function

Passing signals as pre-defined arrays has one major advantage: caching. Even after we restart
the runtime, there won't any recompilation when we pass signal arrays of the same format again.
But sometimes there is a need to trade in a bit of performance for more flexibility. Such use
cases include path-dependency scenarios where signals depend on the previous or current simulation state
such that they cannot be generated in advance. Another use case is to reduce RAM usage: 
the entire indicator and signal generation logic can be encapsulated into a single signal function 
such that there is no need for intermediate arrays anymore. This is useful when a huge
number of parameters needs to be tested, or when the user wants to choose a number of assets
to trade out of a big universe of assets, which usually involves keeping very wide arrays in RAM
but using a signal function would make them redundant.

Let's implement the latest example above but without any arrays!

```pycon
>>> @njit
... def signal_func_nb(c):
...     ts = c.index[c.i]  # (1)!
...     if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 18):  # (2)!
...         return True, False, False, False  # (3)!
...     if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 21):
...         return False, True, False, False
...     if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 22):
...         return False, False, True, False
...     if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 24):
...         return False, False, False, True
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     signal_func_nb=signal_func_nb
... )
>>> pf.assets
Open time
2021-02-18 00:00:00+00:00    0.001940
2021-02-19 00:00:00+00:00    0.001940
2021-02-20 00:00:00+00:00    0.001940
2021-02-21 00:00:00+00:00    0.000000
2021-02-22 00:00:00+00:00   -0.002059
2021-02-23 00:00:00+00:00   -0.002059
2021-02-24 00:00:00+00:00    0.000000
Freq: D, dtype: float64
```

1. `c.index` returns an array with timestamps in the nanosecond format while `c.i`
returns the current row. By applying the latter on the former, we can get the current timestamp.
2. Use [matches_date_nb](/api/utils/datetime_nb/#vectorbtpro.utils.datetime_nb.matches_date_nb)
to check whether the current day matches a date
3. Signal functions must return a set of direction-aware signals

We've switched a vectorized logic in favor of an iterative logic, which is usually more verbose
but also much more flexible and similar to the format used in most open-source backtesting frameworks.
But that doesn't mean that we have to define everything iteratively, we can still
pass one or more arrays and make decisions based on them. All we have to do is to make the signal
function accept arrays as positional arguments and select one element out of them at each time step
to generate the four signals, and then pass the actual arrays to FS as a tuple using `signal_args`.
Just note that any array-like object must be a NumPy array since Numba doesn't understand Pandas.

```pycon
>>> @njit
... def signal_func_nb(c, entries, exits, short_entries, short_exits):
...     long_entry = entries[c.i]
...     long_exit = exits[c.i]
...     short_entry = short_entries[c.i]
...     short_exit = short_exits[c.i]
...     return long_entry, long_exit, short_entry, short_exit

>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         pd.Series([X, O, O, O, O, O, O]).values,
...         pd.Series([O, O, O, X, O, O, O]).values,
...         pd.Series([O, O, O, O, X, O, O]).values,
...         pd.Series([O, O, O, O, O, O, X]).values
...     )
... )
>>> pf.assets
Open time
2021-02-18 00:00:00+00:00    0.001940
2021-02-19 00:00:00+00:00    0.001940
2021-02-20 00:00:00+00:00    0.001940
2021-02-21 00:00:00+00:00    0.000000
2021-02-22 00:00:00+00:00   -0.002059
2021-02-23 00:00:00+00:00   -0.002059
2021-02-24 00:00:00+00:00    0.000000
Freq: D, dtype: float64
```

But what if we wanted to expand our data to multiple assets? The example above would only
work if each of the arrays is kept as one-dimensional since we only select rows in the signal function.
To make our logic shape-agnostic though, we need to make each of the arrays two-dimensional 
and additionally select the current column in the signal function. But that's not enough:
we also need to take care of broadcasting, which can be flexibly done in the signal function
either manually with [flex_select_auto_nb](/api/base/indexing/#vectorbtpro.base.indexing.flex_select_auto_nb)
or automatically with [select_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.select_elem_nb):

```pycon
>>> @njit
... def signal_func_nb(c, entries, exits, short_entries, short_exits):
...     long_entry = vbt.pf_nb.select_nb(c, entries)
...     long_exit = vbt.pf_nb.select_nb(c, exits)
...     short_entry = vbt.pf_nb.select_nb(c, short_entries)
...     short_exit = vbt.pf_nb.select_nb(c, short_exits)
...     return long_entry, long_exit, short_entry, short_exit

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.to_2d_array(pd.Series([X, O, O, O, O, O, O])),
...         vbt.to_2d_array(pd.Series([O, O, O, X, O, O, O])),
...         vbt.to_2d_array(pd.Series([O, O, O, O, X, O, O])),
...         vbt.to_2d_array(pd.Series([O, O, O, O, O, O, X]))
...     )
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.001940  0.051557
2021-02-19 00:00:00+00:00  0.001940  0.051557
2021-02-20 00:00:00+00:00  0.001940  0.051557
2021-02-21 00:00:00+00:00  0.000000  0.000000
2021-02-22 00:00:00+00:00 -0.002059 -0.056080
2021-02-23 00:00:00+00:00 -0.002059 -0.056080
2021-02-24 00:00:00+00:00  0.000000  0.000000
```

Our strategy can now be applied on an arbitrary number of columns, great! But even this isn't
the most flexible design :face_exhaling: What if the user provides a signal array that doesn't
have the same number of rows as the data? If bound checking is enabled, we would get the "index is 
out of bounds" error since the signal function would attempt to select an element that simply doesn't exist.
To make any array broadcast against the data prior to the simulation, we can define it in the
dictionary `broadcast_named_args`, and then use a template to substitute its name by the 
broadcasted array in `signal_args`:

```pycon
>>> entries = pd.Series({pd.Timestamp("2021-02-18 00:00:00+00:00"): True})
>>> exits = pd.Series({pd.Timestamp("2021-02-21 00:00:00+00:00"): True})
>>> short_entries = pd.Series({pd.Timestamp("2021-02-22 00:00:00+00:00"): True})
>>> short_exits = pd.Series({pd.Timestamp("2021-02-24 00:00:00+00:00"): True})

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("entries"),  # (1)!
...         vbt.Rep("exits"),
...         vbt.Rep("short_entries"),
...         vbt.Rep("short_exits")
...     ),
...     broadcast_named_args=dict(
...         entries=entries,
...         exits=exits,
...         short_entries=short_entries,
...         short_exits=short_exits
...     )
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
2021-02-18 00:00:00+00:00  0.001940  0.051557
2021-02-19 00:00:00+00:00  0.001940  0.051557
2021-02-20 00:00:00+00:00  0.001940  0.051557
2021-02-21 00:00:00+00:00  0.000000  0.000000
2021-02-22 00:00:00+00:00 -0.002059 -0.056080
2021-02-23 00:00:00+00:00 -0.002059 -0.056080
2021-02-24 00:00:00+00:00  0.000000  0.000000
```

1. `entries` is recognized as a key in `broadcast_named_args`

Our setup now behaves the same way as the built-in arguments `entries`, `exits`, `short_entries`,
and `short_exits` :magic_wand: We don't even have to convert them to NumPy arrays anymore since this
is being taken care of automatically by the broadcaster. It allows us also to use index dictionaries
and other hot broadcasting features:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("entries"),
...         vbt.Rep("exits"),
...         vbt.Rep("short_entries"),
...         vbt.Rep("short_exits")
...     ),
...     broadcast_named_args=dict(
...         entries=      vbt.index_dict({"2021-02-18": True, "_def": False}),
...         exits=        vbt.index_dict({"2021-02-21": True, "_def": False}),
...         short_entries=vbt.index_dict({"2021-02-22": True, "_def": False}),
...         short_exits=  vbt.index_dict({"2021-02-24": True, "_def": False})
...     )
... )
```

But we're rarely interested in backtesting signals on some fixed dates. Let's create a signal
function that yields a long entry signal when there is an above-crossover and a short entry signal
when there is a below-crossover of two moving average arrays. In addition, we'll parametrize this strategy
by introducing a flexible parameter `wait` that controls the number of bars after which a signal should be 
placed after a crossover has been detected; if an opposite crossover happens during that time, 
the signal gets canceled, thus `wait` also acts as a confirmation period. The parameter will broadcast
together with data to be able to define it per row, column, and element.

```pycon
>>> @njit
... def signal_func_nb(c, fast_sma, slow_sma, wait):
...     curr_wait = vbt.pf_nb.select_nb(c, wait)  # (1)!
...     i_wait = c.i - curr_wait  # (2)!
...     if i_wait < 0:  # (3)!
...         return False, False, False, False
...
...     if vbt.nb.iter_crossed_above_nb(fast_sma, slow_sma, i_wait, c.col):  # (4)!
...         cross_confirmed = True
...         for j in range(i_wait + 1, c.i + 1):  # (5)!
...             if not vbt.nb.iter_above_nb(fast_sma, slow_sma, j, c.col):
...                 cross_confirmed = False
...                 break
...         if cross_confirmed:
...             return True, False, False, False
...
...     if vbt.nb.iter_crossed_below_nb(fast_sma, slow_sma, i_wait, c.col):  # (6)!
...         cross_confirmed = True
...         for j in range(i_wait + 1, c.i + 1):
...             if not vbt.nb.iter_below_nb(fast_sma, slow_sma, j, c.col):
...                 cross_confirmed = False
...                 break
...         if cross_confirmed:
...             return False, False, True, False
...
...     return False, False, False, False

>>> fast_sma = data.run("sma", 20, short_name="fast_sma").real
>>> slow_sma = data.run("sma", 50, short_name="slow_sma").real
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("fast_sma"),
...         vbt.Rep("slow_sma"),
...         vbt.Rep("wait")
...     ),
...     broadcast_named_args=dict(
...         fast_sma=fast_sma,
...         slow_sma=slow_sma,
...         wait=0  # (7)!
...     )
... )
>>> pf.orders.count()
fast_sma_timeperiod  slow_sma_timeperiod  symbol 
20                   50                   BTCUSDT    37
                                          ETHUSDT    29
Name: count, dtype: int64
```

1. Since the confirmation period is an array-like parameter, get the value defined for the current row and column
2. Get the row where the crossover should have taken place
3. If the current period is smaller than the confirmation period, return no signal
4. Check whether the faster SMA has crossed the slower SMA at the current bar
5. If true, loop over the confirmation period (including the current bar) and check whether the faster 
SMA has always stayed above the slower SMA for confirmation
6. The same for the below-crossover
7. Waiting period is a flexible parameter, that is, it must broadcast together with other arrays

To confirm that our strategy has generated a correct number of orders, let's 
get the total number of crossover signals manually:

```pycon
>>> n_crossed_above = fast_sma.vbt.crossed_above(slow_sma).sum()
>>> n_crossed_below = fast_sma.vbt.crossed_below(slow_sma).sum()
>>> n_crossed_above + n_crossed_below
fast_sma_timeperiod  slow_sma_timeperiod  symbol 
20                   50                   BTCUSDT    37
                                          ETHUSDT    29
dtype: int64
```

To demonstrate the full power of the vectorbt's broadcaster, let's test the confirmation period
of 0, 1, 7, and 30 bars by wrapping the parameter with [Param](/api/utils/params/#vectorbtpro.utils.params.Param):

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("fast_sma"),
...         vbt.Rep("slow_sma"),
...         vbt.Rep("wait")
...     ),
...     broadcast_named_args=dict(
...         fast_sma=fast_sma,
...         slow_sma=slow_sma,
...         wait=vbt.Param([0, 1, 7, 30])
...     )
... )
>>> pf.orders.count()
wait  fast_sma_timeperiod  slow_sma_timeperiod  symbol 
0     20                   50                   BTCUSDT    37
                                                ETHUSDT    29
1     20                   50                   BTCUSDT    35
                                                ETHUSDT    29
7     20                   50                   BTCUSDT    33
                                                ETHUSDT    27
30    20                   50                   BTCUSDT    12
                                                ETHUSDT    14
Name: count, dtype: int64
```

We can observe how the number of orders gradually decreases with a longer confirmation period.
If you love vectorbt for its performance, you might argue that even though the number of crossovers
is low, having the second loop is not exactly optimal performance-wise: we can rewrite the logic to 
iterate over the data just once. For this, we need to introduce a temporary array that stores
the index of the latest crossover that has been confirmed so far, and once the confirmation
period is fully over, we can issue a signal. This use case is a perfect example on how to temporarily
store and then share data between multiple signal function calls!

The temporary array that we will create will be a one-dimensional NumPy array where
the latest crossover index is stored per column. We could have also used a regular typed list but 
remember that NumPy arrays enjoy superiority in Numba. Why per column and not just an array with one value? 
This would work for ungrouped portfolios where columns are processed strictly one after another,
but in a case where the portfolio is grouped, columns are processed in a zigzag manner inside their groups,
thus we should always structure our temporary data per column to be on a safe side. Another challenge
is the creation of such an array: how do we know the number of columns in advance? Thankfully,
we can use a template!

```pycon
>>> @njit
... def signal_func_nb(c, fast_sma, slow_sma, wait, temp_coi):  # (1)!
...     if temp_coi[c.col] != -1:  # (2)!
...         crossed_above = vbt.nb.iter_crossed_above_nb(
...             fast_sma, slow_sma, temp_coi[c.col], c.col)
...         crossed_below = vbt.nb.iter_crossed_below_nb(
...             fast_sma, slow_sma, temp_coi[c.col], c.col)
...         if crossed_above:  # (3)!
...             if not vbt.pf_nb.iter_above_nb(c, fast_sma, slow_sma):  # (4)!
...                 temp_coi[c.col] = -1
...         if crossed_below:
...             if not vbt.pf_nb.iter_below_nb(c, fast_sma, slow_sma):
...                 temp_coi[c.col] = -1
...
...     curr_wait = vbt.pf_nb.select_nb(c, wait)
...     if temp_coi[c.col] != -1:  # (5)!
...         if c.i - temp_coi[c.col] == curr_wait:
...             if crossed_above:
...                 temp_coi[c.col] = -1
...                 return True, False, False, False
...             if crossed_below:
...                 temp_coi[c.col] = -1
...                 return False, False, True, False
...     else:  # (6)!
...         if vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma):
...             if curr_wait == 0:
...                 return True, False, False, False
...             temp_coi[c.col] = c.i
...         if vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma):
...             if curr_wait == 0:
...                 return False, False, True, False
...             temp_coi[c.col] = c.i
...
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("fast_sma"),
...         vbt.Rep("slow_sma"),
...         vbt.Rep("wait"),
...         vbt.RepEval("np.full(wrapper.shape_2d[1], -1)")  # (7)!
...     ),
...     broadcast_named_args=dict(
...         fast_sma=fast_sma,
...         slow_sma=slow_sma,
...         wait=vbt.Param([0, 1, 7, 30])
...     )
... )
>>> pf.orders.count()
wait  fast_sma_timeperiod  slow_sma_timeperiod  symbol 
0     20                   50                   BTCUSDT    37
                                                ETHUSDT    29
1     20                   50                   BTCUSDT    35
                                                ETHUSDT    29
7     20                   50                   BTCUSDT    33
                                                ETHUSDT    27
30    20                   50                   BTCUSDT    12
                                                ETHUSDT    14
Name: count, dtype: int64
```

1. The temporary array is taken as a regular argument
2. Check whether there is a crossover index stored under this column
3. If true, check the type of the crossover and whether it can still be confirmed. 
If it cannot be confirmed at this bar, remove the index.
4. Use generic iterative functions (starting with `vbt.nb.iter_`, see [generic.iter_](/api/generic/nb/iter_/)) 
when an operation must be performed on a custom row/column, and portfolio iterative functions 
(starting with `vbt.pf_nb.iter_`, see [portfolio.iter_](/api/portfolio/nb/iter_/)) to perform an 
operation on the current row/column taken from the context
5. If there is still an index in the temporary array, the crossover has been confirmed so far,
thus check if the confirmation period is over, and if true, return the signal
6. If there is no crossover index, check whether there is a crossover at this bar,
and if true, store it inside the temporary array and move on. If the confirmation period
is zero, return the signal right away.
7. Create the temporary array `temp_coi`. Use an evaluation template to run a code as an expression 
after all the arrays have been broadcasted and the final shape has been established. 
Use `wrapper` to access the shape information.

The code above isn't even complex: it would take the same amount of code to define this logic
in a conventional backtesting software. The only difference is that vectorbt relies on functional
programming whereas other frameworks rely on object-oriented programming where functions
such as `crossed_above_nb` are methods of the current backtesting instance (`self.crossed_above()`) 
and variables such as `temp_coi` are attributes of that instance (`self.temp_coi`).

### Conflicts

When signals are generated in an automated way, it often happens that multiple signals 
of the same type come one after another, or multiple signals of different types appear at the same bar.
The first case is effectively solved by the FS method by taking into consideration only the first signal 
while ignoring others (unless [accumulation](#accumulation) is turned on), as demonstrated in the following 
example where we're repeatedly issuing a long entry signal:

```pycon
>>> pf = vbt.Portfolio.from_signals(sub_data, entries=True)  # (1)!
>>> pf.asset_flow
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00000  0.000000
2021-02-20 00:00:00+00:00  0.00000  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

1. When signals are provided as a single value, the value gets broadcast against the entire data shape
such that it appears under each single row and column

But what if we start issuing a long exit signal at the same time?

```pycon
>>> pf = vbt.Portfolio.from_signals(sub_data, entries=True, exits=True)
>>> pf.asset_flow
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      0.0      0.0
2021-02-19 00:00:00+00:00      0.0      0.0
2021-02-20 00:00:00+00:00      0.0      0.0
2021-02-21 00:00:00+00:00      0.0      0.0
2021-02-22 00:00:00+00:00      0.0      0.0
2021-02-23 00:00:00+00:00      0.0      0.0
2021-02-24 00:00:00+00:00      0.0      0.0
```

We can see that the simulator simply ignored conflicting signals. But sometimes, we may
want to prefer one signal type over another. In the example above, we have a so-called "long signal 
conflict", the resolution of which is controlled by the argument `upon_long_conflict`
of the type [ConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.ConflictMode). 
For example, if long entries are more important than long exits:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data, 
...     entries=True, 
...     exits=True,
...     upon_long_conflict="entry"
... )
>>> pf.asset_flow
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00  0.00000  0.000000
2021-02-20 00:00:00+00:00  0.00000  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

What about use cases where we allow both directions for an exit signal to become a short entry signal?
If so, we would have a so-called "signal direction conflict", the resolution of which is controlled 
by the argument `upon_dir_conflict` of the type [DirectionConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.DirectionConflictMode). 
Let's prefer short signals over long signals, in any direction:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data, 
...     entries=True, 
...     exits=True,
...     direction="both",
...     upon_dir_conflict="short"
... )
>>> pf.asset_flow
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00 -0.00194 -0.051557
2021-02-19 00:00:00+00:00  0.00000  0.000000
2021-02-20 00:00:00+00:00  0.00000  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

We can see that both orders have become sell orders. Let's combine both use cases and
apply our knowledge to the scenario where all four signals are set! We will open a long position
at the first bar, while at other bars only the one signal that's opposite to the current
position will win and reverse the position:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data, 
...     entries=True, 
...     exits=vbt.index_dict({0: False, "_def": True}),
...     short_entries=vbt.index_dict({0: False, "_def": True}),
...     short_exits=vbt.index_dict({0: False, "_def": True}),
...     upon_long_conflict="entry",
...     upon_short_conflict="entry",
...     upon_dir_conflict="opposite",
...     upon_opposite_entry="reverse"
... )
>>> pf.asset_flow
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.001940  0.051557
2021-02-19 00:00:00+00:00 -0.003880 -0.103114
2021-02-20 00:00:00+00:00  0.003884  0.105377
2021-02-21 00:00:00+00:00 -0.003889 -0.107641
2021-02-22 00:00:00+00:00  0.004127  0.117085
2021-02-23 00:00:00+00:00 -0.004366 -0.126528
2021-02-24 00:00:00+00:00  0.004297  0.122982
```

Great, we forced vectorbt to reverse the position at each bar :cold_face:

## Orders

As we've learned, signals are just another level of abstraction over orders; they control the timing
and direction of orders. But how do we specify the parameters of a typical order a signal should be 
converted into? Identically to [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders), 
the class method [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
takes a variety of order-related arguments; in fact, it takes all the arguments that can be 
found as a field in the class [Order](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Order). 
If any argument remains `None`, the method takes the default value defined in the
[portfolio settings](/api/_settings/#vectorbtpro._settings.portfolio). For example, the default
size is `np.inf`, meaning that each signal instructs the simulator to use up the entire capital. 
Also, each order-related argument is array-like and broadcasts together with signals. This way, 
by setting an argument to a single value, we can enable that value for each signal. Let's make each 
entry signal order $1 worth of each asset by tweaking the size and size type arguments:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     size=1,
...     size_type="value"
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000019  0.000516
2021-02-19 00:00:00+00:00  0.000019  0.000516
2021-02-20 00:00:00+00:00  0.000019  0.000516
2021-02-21 00:00:00+00:00  0.000000  0.000000
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00  0.000000  0.000000
2021-02-24 00:00:00+00:00  0.000000  0.000000
```

As we can see, the simulator ordered `1 / 51552.60 = 0.000019` units of `BTCUSDT` and then closed out 
the position. To have a more granular control over any order information, we can specify the information 
as an array. For example, let's enter a position with 50% of the available cash, close the position, 
then open a new position of the opposite direction with 25% of the available cash, and close the position again:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=      pd.Series([X,   O, O, O, O,    O, O]),
...     exits=        pd.Series([O,   O, O, X, O,    O, O]),
...     short_entries=pd.Series([O,   O, O, O, X,    O, O]),
...     short_exits=  pd.Series([O,   O, O, O, O,    O, X]),
...     size=         pd.Series([0.5, 0, 0, 0, 0.25, 0, 0]),
...     size_type="valuepercent",  # (1)!
... )
>>> pf.asset_value / pf.value  # (2)!
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.500000  0.500000  << long entry of 50%
2021-02-19 00:00:00+00:00  0.520256  0.501976
2021-02-20 00:00:00+00:00  0.519967  0.496546
2021-02-21 00:00:00+00:00  0.000000  0.000000  << long exit
2021-02-22 00:00:00+00:00 -0.250000 -0.250000  << short entry of 25%
2021-02-23 00:00:00+00:00 -0.220680 -0.215853
2021-02-24 00:00:00+00:00  0.000000  0.000000  << short exit
```

1. Treat the size as a percentage of the total portfolio value
2. Get the allocation at each bar, that is, the ratio of the asset value in respect to 
the total portfolio value

Thanks to the vectorbt's powerful broadcasting mechanism, we can backtest arbitrary 
configurations with a couple lines of code. Below, we're testing three different
mutual configurations of the size and size type arguments:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     size=     vbt.Param([np.inf,   1,       0.5], product_idx=0),  # (1)!
...     size_type=vbt.Param(["amount", "value", "valuepercent"], product_idx=0)
... )
>>> pf.total_return
size  size_type     symbol 
inf   amount        BTCUSDT    0.113592
                    ETHUSDT   -0.003135
1.0   value         BTCUSDT    0.001136
                    ETHUSDT   -0.000031
0.5   valuepercent  BTCUSDT    0.056796
                    ETHUSDT   -0.001567
Name: total_return, dtype: float64
```

1. Without the same `product_idx`, both parameters would build a Cartesian product

Each configuration is getting applied on the entire set of signal arrays.

### Accumulation

If we wanted to sell 1$ worth of each asset instead of closing out the entire position whenever 
an exit signal is encountered, we need to turn on accumulation:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     size=1,
...     size_type="value",
...     accumulate=True
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000019  0.000516
2021-02-19 00:00:00+00:00  0.000019  0.000516
2021-02-20 00:00:00+00:00  0.000019  0.000516
2021-02-21 00:00:00+00:00  0.000002  0.000000
2021-02-22 00:00:00+00:00  0.000002  0.000000
2021-02-23 00:00:00+00:00  0.000002  0.000000
2021-02-24 00:00:00+00:00  0.000002  0.000000
```

There is a leftover in the column `BTCUSDT` since we made a profit, while the position
has been closed entirely in the column `ETHUSDT` since we made a loss ($1 is worth less than 
at the beginning of the simulation). But there is another implication: whenever the accumulation is turned 
on, the method starts behaving effectively like [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders).
For instance, it treats each signal as an order, irrespective of whether we're in the market or not.
This is best illustrated by the following example, where we're issuing a long entry signal
at each single bar, without and with the accumulation:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=True,
...     size=1,
...     size_type="value",
...     accumulate=vbt.Param([False, True])
... )
>>> pf.asset_flow  # (1)!
accumulate                    False                True          
symbol                      BTCUSDT   ETHUSDT   BTCUSDT   ETHUSDT
Open time                                                        
2021-02-18 00:00:00+00:00  0.000019  0.000516  0.000019  0.000516
2021-02-19 00:00:00+00:00  0.000000  0.000000  0.000018  0.000512
2021-02-20 00:00:00+00:00  0.000000  0.000000  0.000018  0.000523
2021-02-21 00:00:00+00:00  0.000000  0.000000  0.000017  0.000517
2021-02-22 00:00:00+00:00  0.000000  0.000000  0.000018  0.000563
2021-02-23 00:00:00+00:00  0.000000  0.000000  0.000020  0.000634
2021-02-24 00:00:00+00:00  0.000000  0.000000  0.000020  0.000616
```

1. Calculate and print the asset flow at each bar, that is, the absolute amount of units 
bought (positive) or sold (negative)

We can see that without the accumulation, only the first signal is getting executed while all the
signals that follow are getting ignored since we're already in a position. With the accumulation
though, each signal gets executed irrespective of the current position. This allows for pyramiding
and other trading schemes that require binary position requirements to be lifted.

### Size types

There is a variety of size types supported. To get the full list, refer to the enumerated type 
[SizeType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SizeType). Each size type can be provided 
either as a (case-insensitive) string representing a field name, or an integer representing a value, 
such that the arguments `size_type="value"` and `size_type=SizeType.Value` will behave identically.
Most size types will be internally converted into the size type `Amount`, which is the absolute
amount of units to order. The conversion is mostly done using the valuation price `val_price`,
which defaults to the order price and is meant to be the latest price at the point of decision-making.

!!! note
    When working with Numba-compiled functions directly, only the integer format is supported.

But not all size types are supported in [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals): 
any target size that defines a target, such as `TargetAmount`, `TargetValue`, and `TargetPercent(100)`, 
cannot be safely used since the final order size may contradict the signal. For example: if we're in a 
position of 10 units, and we've issued an entry signal with the size of 3 units and the size type of `TargetAmount`, 
the actual order will be a sell order of the size 7, which is opposite to the direction of the signal that 
we've issued. Additionally, the size type `Percent` cannot be used in certain circumstances when both 
directions are allowed, such as when reversing a position, because such a percentage cannot be simply "flipped".

### Size granularity

Size is always represented as a (usually 64-bit) floating-point number and the entire simulation 
logic of vectorbt is also based on this number format. But, as you might have heard, floating
numbers aren't exactly ideal for monetary calculations because the floating-point arithmetic 
often causes a loss of precision. Since Numba doesn't allow us to use fixed-point numbers,
vectorbt is forced to use floating-point numbers and apply a range of tricks to compare 
them with a certain confidence, such as by using 
[numpy.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) and 
[numpy.round_](https://numpy.org/doc/stable/reference/generated/numpy.round_.html).

But what about use cases where the size needs to be an integer, such as when trading stocks? 
For this, we can use the argument `size_granularity`, which will round off the final absolute 
size to some number of decimal places. As a rule of thumb: use `1` for whole shares, `0.001` for 
fractional shares, and some custom value for crypto. For example, Binance provides the step size 
of each trading pair, which can be directly used as `size_granularity`.

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     size_granularity=vbt.Param([1, 0.001]),
...     init_cash=1000_000
... )
>>> pf.asset_flow
size_granularity            1.000           0.001         
symbol                    BTCUSDT ETHUSDT BTCUSDT  ETHUSDT
Open time                                                 
2021-02-18 00:00:00+00:00    19.0   515.0  19.397  515.567
2021-02-19 00:00:00+00:00     0.0     0.0   0.000    0.000
2021-02-20 00:00:00+00:00     0.0     0.0   0.000    0.000
2021-02-21 00:00:00+00:00   -19.0  -515.0 -19.397 -515.566
2021-02-22 00:00:00+00:00     0.0     0.0   0.000    0.000
2021-02-23 00:00:00+00:00     0.0     0.0   0.000    0.000
2021-02-24 00:00:00+00:00     0.0     0.0   0.000    0.000
```

!!! info
    Even though the traded size appears as an integer, it's still represented by a float.

### Price

By default, vectorbt executes an order right away using the current closing price.
This behavior can be changed by tweaking the argument `price`, which can either take
a price array, or a price option of the type [PriceType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceType).
If we take a look into the portfolio settings, we'll find that `price` is set to the string `"close"`,
which upon running is getting translated into the option `PriceType.Close`. This option
is just an alias for the value `np.inf` (positive infinity). What does an infinity do here?
Since an order price must be defined within the price bounds of a bar, the negative and positive 
infinity represent the opening and closing price respectively (see [Price resolution](/documentation/portfolio/#price-resolution)); 
both can be used inside arrays. The other two options `NextOpen` and `NextClose` 
are standalone options and cannot be used inside arrays because they involve other arguments.

!!! important
    Which option to choose depends on which price you used to generate your signals.
    Mostly, signals are generated and executed using the same closing price. To account for potential 
    time gaps between signal generation and execution, use the next open or close, or shift 
    the signals manually. If you generated the signals using the opening price or any other price 
    preceding it, you can also use the option `"open"` and a bit of slippage.

Let's execute entries using open and exits using close. Remember that each order-related argument
is defined for all signals types: long entries and exits, and short entries and exits. That means
that there isn't any argument that defines price specifically for exits, otherwise, the number of
arguments would skyrocket. To make any argument value valid only for a subset of signal types, 
we should set it by using the signal types a mask:

```pycon
>>> price = sub_data.symbol_wrapper.fill()  # (1)!
>>> entries = pd.Series([X, O, O, O, O, O, O]).vbt.broadcast_to(price)  # (2)!
>>> exits =   pd.Series([O, O, O, X, O, O, O]).vbt.broadcast_to(price)
>>> price[entries] = sub_data.open  # (3)!
>>> price[exits] = sub_data.close

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=entries,
...     exits=exits,
...     price=price
... )

>>> pf.orders.price.to_pd() == sub_data.open  # (4)!
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00     True     True
2021-02-19 00:00:00+00:00    False    False
2021-02-20 00:00:00+00:00    False    False
2021-02-21 00:00:00+00:00    False    False
2021-02-22 00:00:00+00:00    False    False
2021-02-23 00:00:00+00:00    False    False
2021-02-24 00:00:00+00:00    False    False

>>> pf.orders.price.to_pd() == sub_data.close  # (5)!
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00    False    False
2021-02-19 00:00:00+00:00    False    False
2021-02-20 00:00:00+00:00    False    False
2021-02-21 00:00:00+00:00     True     True
2021-02-22 00:00:00+00:00    False    False
2021-02-23 00:00:00+00:00    False    False
2021-02-24 00:00:00+00:00    False    False
```

1. Create a new price array of the same shape as the data
2. Define signals and broadcast them to the price array to be able to use them as a mask
3. Set any element in price that falls under an entry signal to the opening price
4. Generate a mask of filled orders which price is open (such a comparison works only if slippage is zero!)
5. Generate a mask of filled orders which price is close

Let's order using the opening price of the next bar instead:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     price="nextopen"
... )
>>> pf.orders.price.to_pd() == sub_data.open
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00    False    False
2021-02-19 00:00:00+00:00     True     True
2021-02-20 00:00:00+00:00    False    False
2021-02-21 00:00:00+00:00    False    False
2021-02-22 00:00:00+00:00     True     True
2021-02-23 00:00:00+00:00    False    False
2021-02-24 00:00:00+00:00    False    False
```

As we can see, the simulator waited one bar and then executed each signal using the opening price.
This is one of the safest approaches when it comes to backtesting because now we have the freedom
of running indicators on any price, and we don't have to worry about the look-ahead bias during the 
execution anymore. The most bulletproof approach would be using the next close because the difference between
the previous close and the next open is usually negligible.

### Shifting

We could have done the same as above by shifting the signal arrays by one bar manually:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]).vbt.signals.fshift(),
...     exits=  pd.Series([O, O, O, X, O, O, O]).vbt.signals.fshift(),
...     price="open"  # (1)!
... )
>>> pf.orders.price.to_pd() == sub_data.open
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00    False    False
2021-02-19 00:00:00+00:00     True     True
2021-02-20 00:00:00+00:00    False    False
2021-02-21 00:00:00+00:00    False    False
2021-02-22 00:00:00+00:00     True     True
2021-02-23 00:00:00+00:00    False    False
2021-02-24 00:00:00+00:00    False    False
```

1. When shifting manually, use open/close of the current bar

And this is one of the most underrated features of vectorbt: since we're working with array
data, we can shift the data to make any current bar use information from the past. In the example
above, after we've forward-shifted the signal arrays, the long entry signal on the 18th
February has moved to the 19th February while the index has stayed the same. Thus, the surrounding price 
information in form of OHLC (the arguments `open`, `high`, `low`, and `close`) must be fixed
and should never be shifted! That also means that we not only have to shift the signals,
but any information those signals link to, for example, the order direction and price:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=  pd.Series([X, O, O, O, X, O, O]).vbt.fshift(1, False),
...     exits=    pd.Series([O, O, O, X, O, O, X]).vbt.fshift(1, False),
...     direction=pd.Series([L, L, L, L, S, S, S]).vbt.fshift(1, -1)
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000000  0.000000
2021-02-19 00:00:00+00:00  0.001789  0.051151
2021-02-20 00:00:00+00:00  0.001789  0.051151
2021-02-21 00:00:00+00:00  0.001789  0.051151
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00 -0.001979 -0.057624
2021-02-24 00:00:00+00:00 -0.001979 -0.057624
```

!!! hint
    As a rule of thumb: if an argument is signal-anchored, it should be shifted as well.
    If an argument is date-anchored though, it should stay the same.

To reduce the burden of shifting manually, vectorbt can do it for us automatically!
For this, it requires from us providing the argument `from_ago`, which represents how far ago was the 
bar from which all the signal and order information should be taken. For example, when `from_ago=1`, 
the related information is taken from the previous bar:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=  pd.Series([X, O, O, O, X, O, O]),
...     exits=    pd.Series([O, O, O, X, O, O, X]),
...     direction=pd.Series([L, L, L, L, S, S, S]),
...     from_ago=1
... )
>>> pf.assets
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000000  0.000000
2021-02-19 00:00:00+00:00  0.001789  0.051151
2021-02-20 00:00:00+00:00  0.001789  0.051151
2021-02-21 00:00:00+00:00  0.001789  0.051151
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00 -0.001979 -0.057624
2021-02-24 00:00:00+00:00 -0.001979 -0.057624
```

This argument can also be provided as an array.

### Slippage

By default, an order is executed as a market order. This can be seen in the portfolio settings,
under the key `order_type`. Market orders are transactions meant to execute as quickly as possible 
at the current market price, which is always `price` (the closing price by default). In reality though,
the price at which an order is executed often does not match the price at which it was requested.
To take into account this discrepancy, we need to introduce a slippage. Assuming the trading volume 
is high, we should see less slippage. And when the trading volume is slow, we should expect to 
see more slippage. An optimal slippage can be calculated from order book data 
(see [this blog](https://www.hodlbot.io/blog/an-analysis-of-slippage-on-the-binance-exchange)).

But let for a second assume that the average slippage is 0.5%. Using it together with the default
price is generally a bad idea since the closing price is meant to be the latest price seen at each bar.
To make our simulation more realistic, we need to apply it on the next open instead:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     price="nextopen",
...     slippage=0.005  # (1)!
... )
>>> pf.orders.price.to_pd()  # (2)!
symbol                         BTCUSDT     ETHUSDT
Open time                                         
2021-02-18 00:00:00+00:00          NaN         NaN
2021-02-19 00:00:00+00:00  51810.37305  1948.60455
2021-02-20 00:00:00+00:00          NaN         NaN
2021-02-21 00:00:00+00:00          NaN         NaN
2021-02-22 00:00:00+00:00  57125.28825  1923.87230
2021-02-23 00:00:00+00:00          NaN         NaN
2021-02-24 00:00:00+00:00          NaN         NaN

>>> pf.orders.price.to_pd() / sub_data.open  # (3)!
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00    1.005    1.005
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00    0.995    0.995
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. 0.01 is 1%
2. Get the filled order price
3. Get the filled order price in relation to the opening price

We can see that the slippage increased the price by 0.5% when buying and decreased
the price by 0.5% when selling. Introducing a slippage will always result in a fixed price penalty,
thus the slippage should always reflect the average penalty that is being recorded in the market for
transactions of this magnitude. Since slippage isn't static but depending on various factors, 
we can provide it as an array.

### Limit orders

To help eliminate or reduce slippage, traders use limit orders instead of market orders. A limit order 
only fills at the price we want, or better. Unlike a market order, it won't fill at a worse price.
But there is a catch: while the price is guaranteed, the filling of the order is not, and limit orders 
will not be executed unless the asset price meets the order qualifications. If the asset does not reach 
the specified price, the order is not filled, and we may miss out on the trading opportunity. So,
what happens when we execute our signals using limit orders? Let's switch the default order type
by setting the argument `order_type` to `"limit"`:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     exits=  pd.Series([O, O, O, X, O, O, O]),
...     order_type="limit"
... )
>>> pf.orders.price.to_pd()
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00       NaN      NaN
2021-02-19 00:00:00+00:00  51552.60  1938.91
2021-02-20 00:00:00+00:00       NaN      NaN
2021-02-21 00:00:00+00:00       NaN      NaN
2021-02-22 00:00:00+00:00  57412.35  1933.54
2021-02-23 00:00:00+00:00       NaN      NaN
2021-02-24 00:00:00+00:00       NaN      NaN
```

Since the default order price is the closing price, the simulator registered it as the target limit
price and skipped the entry bar. At the next bar, the simulator checked whether the limit price
has been hit by comparing it against the full candle. As we can see above, each of the target prices
could be satisfied already at the next bar, which is quite similar to using the next opening price
as the order execution price. But such a rapid match isn't always the case: what if we wanted
to enter a short trade on the 22nd February using the previous high as the limit price?

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, O, O, X, O, O]),
...     direction="shortonly",
...     price=sub_data.high.vbt.fshift(),  # (1)!
...     order_type="limit"
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Shift the high price by one bar to use the previous high price

Suddenly, no limit order could be filled as the same price or higher (we're selling) couldn't be 
found at any time during or after the 22nd February. We have now an order that can potentially 
remain pending forever. How do we limit its lifetime? There are various possibilities.

#### Time in force

Time-in-force orders can be created using the argument `limit_tif`, which expects a time delta, in any format 
that can be translated into `np.timedelta64` and then to a 64-bit integer representing nanoseconds, 
that is, a string, `pd.Timedelta`, `datetime.timedelta`, `np.timedelta64`, or an integer. The time-in-force 
specification starts counting at the **beginning** of the entry bar (even if the limit order has been issued 
at the end of the bar) and will be checked at the beginning of each bar starting from the second one.
Let's create a buy limit order on the 20th February using the previous low price:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit"
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN   1891.0
2021-02-22 00:00:00+00:00  50710.2      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

We can see that the limit order in the column `BTCUSDT` was executed in two bars
while the limit order in the column `ETHUSDT` was executed in just one bar.
Below, we're testing a TIF option passed as a frequency string, which works only outside of arrays:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_tif="2d"  # (1)!
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN   1891.0
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Gets translated into a time delta using 
[freq_to_timedelta64](/api/utils/datetime_/#vectorbtpro.utils.datetime_.freq_to_timedelta64)

Why is there is no order in `BTCUSDT`? Because 2 days from `2021-02-20 00:00:00` is `2021-02-22 00:00:00`,
which is the timestamp at which the order gets cancelled, and since each timestamp represents the 
opening time of a bar, the pending order doesn't exist during the fifth bar anymore. To provide
a TIF specification inside an array or to test multiple configurations, instead of a string,
we must use either a Pandas or NumPy format, or a total duration in nanoseconds.
Let's test all formats at once!

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_tif=vbt.Param([
...         -1,  # (1)!
...         pd.Timedelta(days=1),  # (2)!
...         2 * np.timedelta64(86400000000000),  # (3)!
...         3 * vbt.dt_nb.d_ns  # (4)!
...     ], keys=["none", "1 days", "2 days", "3 days"])
... )
>>> pf.orders.price.to_pd()
limit_tif                     none          1 days          2 days          \\
symbol                     BTCUSDT ETHUSDT BTCUSDT ETHUSDT BTCUSDT ETHUSDT   
Open time                                                                    
2021-02-18 00:00:00+00:00      NaN     NaN     NaN     NaN     NaN     NaN   
2021-02-19 00:00:00+00:00      NaN     NaN     NaN     NaN     NaN     NaN   
2021-02-20 00:00:00+00:00      NaN     NaN     NaN     NaN     NaN     NaN   
2021-02-21 00:00:00+00:00      NaN  1891.0     NaN     NaN     NaN  1891.0   
2021-02-22 00:00:00+00:00  50710.2     NaN     NaN     NaN     NaN     NaN   
2021-02-23 00:00:00+00:00      NaN     NaN     NaN     NaN     NaN     NaN   
2021-02-24 00:00:00+00:00      NaN     NaN     NaN     NaN     NaN     NaN   

limit_tif                   3 days          
symbol                     BTCUSDT ETHUSDT  
Open time                                   
2021-02-18 00:00:00+00:00      NaN     NaN  
2021-02-19 00:00:00+00:00      NaN     NaN  
2021-02-20 00:00:00+00:00      NaN     NaN  
2021-02-21 00:00:00+00:00      NaN  1891.0  
2021-02-22 00:00:00+00:00  50710.2     NaN  
2021-02-23 00:00:00+00:00      NaN     NaN  
2021-02-24 00:00:00+00:00      NaN     NaN  
```

1. Don't use TIF
2. TIF of 1 day in Pandas format
3. TIF of 2 days in NumPy format
4. TIF of 3 days in integer format (nanoseconds)

!!! note
    NumPy and integer formats are preferred when building large arrays.

There is also a way to specify a number of rows as opposed to a time delta by tweaking the time delta 
format (`time_delta_format`), which is represented by the enumerated type 
[TimeDeltaFormat](/api/portfolio/enums/#vectorbtpro.portfolio.enums.TimeDeltaFormat). This is 
required in cases where the index is not datetime-like, such as after splitting the data
for cross-validation. Let's change the time delta format to `Rows` and do the same as above
but now waiting for a specific number of rows to pass:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_tif=vbt.Param(
...         [-1, 1, 2, 3], 
...         keys=["none", "1 rows", "2 rows", "3 rows"]
...     ),
...     time_delta_format="rows"
... )
```

If we're confident that our index has a fixed frequency (for example, days instead of business days)
and doesn't have gaps, we can also determine the number of rows by dividing time deltas:

```pycon
>>> index_td = vbt.dt.freq_to_timedelta("1 minute")
>>> tif_td = vbt.dt.freq_to_timedelta("1 day")
>>> int(tif_td / index_td)
1440
```

#### Expiration date

Another way to let limit orders expire based on dates and times is by setting an expiration date 
with `limit_expiry`. We've got two meaningful options here: setting a frequency at which limit orders 
should expire, or providing a datetime-like array that can incorporate `pd.Timestamp`, `datetime.datetime`,
and `np.datetime64` objects. In the end, the argument will be converted into the 64-bit integer format
representing a Unix time (total nanoseconds since 1970). Let's play with the first option first
by making limit orders behave like day orders! By receiving a frequency, the simulation method
will determine the right bound of each timestamp based by using the method
[ArrayWrapper.get_period_ns_index](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.get_period_ns_index):

```pycon
>>> sub_data.symbol_wrapper.get_period_ns_index("1d")
array([1613692800000000000, 1613779200000000000, 1613865600000000000,
       1613952000000000000, 1614038400000000000, 1614124800000000000,
       1614211200000000000])
```

But since our data is already daily, let's make a pending limit order expire at
the end of the week in which it was created:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_expiry="W-MON"  # (1)!
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN   1891.0
2021-02-22 00:00:00+00:00  50710.2      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN

>>> sub_data.symbol_wrapper.index.to_period("W-MON")  # (2)!
PeriodIndex(['2021-02-16/2021-02-22', '2021-02-16/2021-02-22',
             '2021-02-16/2021-02-22', '2021-02-16/2021-02-22',
             '2021-02-16/2021-02-22', '2021-02-23/2021-03-01',
             '2021-02-23/2021-03-01'],
            dtype='period[W-MON]', name='Open time')
```

1. See [Anchored offsets](https://pandas.pydata.org/docs/user_guide/timeseries.html#anchored-offsets)
2. User warning "Converting to PeriodArray/Index representation will drop timezone information" can be ignored

Both orders could be executed because the dates `2021-02-20`, `2021-02-21`, and `2021-02-22`
belong to the same week `2021-02-16/2021-02-22`. As soon as we had changed the week layout to start 
on a Sunday though, only the first two dates would belong to the same week, thus resulting in the 
expiration of the pending order in the column `BTCUSDT`:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_expiry="W-SUN"
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN   1891.0
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN

>>> sub_data.symbol_wrapper.index.to_period("W-SUN")
PeriodIndex(['2021-02-15/2021-02-21', '2021-02-15/2021-02-21',
             '2021-02-15/2021-02-21', '2021-02-15/2021-02-21',
             '2021-02-22/2021-02-28', '2021-02-22/2021-02-28',
             '2021-02-22/2021-02-28'],
            dtype='period[W-SUN]', name='Open time')
```

We can also build our own `limit_expiry` array. For instance, let's simulate a TIF
of 2 days by relying on the expiration dates alone - the same as passing `limit_tif="2d"`:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([O, O, X, O, O, O, O]),
...     price=sub_data.low.vbt.fshift(),
...     order_type="limit",
...     limit_expiry=vbt.RepEval("""
...     expiry_index = wrapper.index + pd.Timedelta(days=2)
...     expiry_arr = vbt.to_2d_array(expiry_index.values.astype(np.int64))
...     expiry_arr
...     """)  # (1)!
... )
>>> pf.orders.price.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN   1891.0
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Since, after broadcasting, the final index might be different from the original one,
we need to create a template that adds 2 days to the final index and converts it
into a two-dimensional array. Leaving it as one-dimensional will broadcast values per column (!)

!!! note
    Don't attempt to pass a `pd.Index` directly, it will be converted into parameters.
    Use `pd.Series` instead.

#### Conflicts

What happens if the user issues a signal when there is a limit order pending? Since FS can track
only one limit order per column at a time, we need to choose a winner. There are two arguments
that are relevant: `upon_adj_limit_conflict` and `upon_opp_limit_conflict`, both of the type 
[PendingConflictMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PendingConflictMode). 
The first controls the decision process if both orders attempt to go in the same direction,
the second if their directions are opposite. Let's introduce a sell market order on the 21st February
and test various conflict resolution options:

```pycon
>>> all_conflict_modes = vbt.pf_enums.PendingConflictMode._fields  # (1)!
>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     entries=      pd.Series([O, O, X, O, O, O, O]),
...     short_entries=pd.Series([O, O, O, X, O, O, O]),
...     price=sub_data.select("BTCUSDT").low.vbt.fshift(),
...     order_type=vbt.index_dict({2: "limit", "_def": "market"}),  # (2)!
...     upon_opp_limit_conflict=vbt.Param(all_conflict_modes)
... )
>>> pf.orders.price.to_pd()
upon_opp_limit_conflict    KeepIgnore  KeepExecute  CancelIgnore  \\
Open time                                                          
2021-02-18 00:00:00+00:00         NaN          NaN           NaN   
2021-02-19 00:00:00+00:00         NaN          NaN           NaN   
2021-02-20 00:00:00+00:00         NaN          NaN           NaN   
2021-02-21 00:00:00+00:00         NaN     53863.93           NaN   
2021-02-22 00:00:00+00:00     50710.2     50710.20           NaN   
2021-02-23 00:00:00+00:00         NaN          NaN           NaN   
2021-02-24 00:00:00+00:00         NaN          NaN           NaN   

upon_opp_limit_conflict    CancelExecute  
Open time                                 
2021-02-18 00:00:00+00:00            NaN  
2021-02-19 00:00:00+00:00            NaN  
2021-02-20 00:00:00+00:00            NaN  
2021-02-21 00:00:00+00:00       53863.93  
2021-02-22 00:00:00+00:00            NaN  
2021-02-23 00:00:00+00:00            NaN  
2021-02-24 00:00:00+00:00            NaN  
```

1. Get the list with all options
2. Only the first order should be a limit order

As you might have noticed, the first word in the option's name means what to do with the
pending limit order, and the second word means what to do with the user-defined order.
By default, the limit order wins when the user-defined signal is of the same direction (buy and buy,
sell and sell, option `KeepIgnore`) to avoid repeatedly executing similar signals, and the 
user-defined signal wins when both orders are of the opposite direction (buy and sell, sell and 
buy, option `CancelExecute`) to account for a change in the market regime.

#### Delta

The target price of a limit order is taken from the argument `price`. But what if we wanted
to place a limit order at a price some distance (a.k.a. "delta") above or below some another price, 
such as 10% from the current closing price? For this, we would need to know whether a signal
is a buy or sell order beforehand: if it's a buy order, the price should be lower than the closing price, 
and if it's a sell order, the price should be higher than the closing price. To avoid determining
such information in advance and building a price array manually, there is a handy argument 
`limit_delta` that specifies how far away from `price` the target limit price should be.
The format of this argument is controlled by another argument `delta_format` of the type
[DeltaFormat](/api/portfolio/enums/#vectorbtpro.portfolio.enums.DeltaFormat). The default format
is a percentage (`Percent`), such as passing `limit_delta=0.1` will be understood as 10%.
Let's try out different deltas for a buy limit order at the very first bar:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     order_type="limit",
...     limit_delta=vbt.Param([0, 0.1, 0.5])
... )
>>> pf.orders.price.to_pd()
limit_delta                    0.0                0.1               0.5  \\
symbol                     BTCUSDT  ETHUSDT   BTCUSDT   ETHUSDT BTCUSDT   
Open time                                                                 
2021-02-18 00:00:00+00:00      NaN      NaN       NaN       NaN     NaN   
2021-02-19 00:00:00+00:00  51552.6  1938.91       NaN       NaN     NaN   
2021-02-20 00:00:00+00:00      NaN      NaN       NaN       NaN     NaN   
2021-02-21 00:00:00+00:00      NaN      NaN       NaN       NaN     NaN   
2021-02-22 00:00:00+00:00      NaN      NaN       NaN  1745.649     NaN   
2021-02-23 00:00:00+00:00      NaN      NaN  46397.34       NaN     NaN   
2021-02-24 00:00:00+00:00      NaN      NaN       NaN       NaN     NaN   

limit_delta                        
symbol                    ETHUSDT  
Open time                          
2021-02-18 00:00:00+00:00     NaN  
2021-02-19 00:00:00+00:00     NaN  
2021-02-20 00:00:00+00:00     NaN  
2021-02-21 00:00:00+00:00     NaN  
2021-02-22 00:00:00+00:00     NaN  
2021-02-23 00:00:00+00:00     NaN  
2021-02-24 00:00:00+00:00     NaN  
```

As we can see, without a delta, the target limit price is essentially the current closing price.
With a limit delta of 10%, the target limit price becomes `close * (1 - limit_delta)`,
which in our example have only been matched after a few bars. The final limit delta of 50% haven't 
been matched at all since the price window that we chose doesn't have dips of such a magnitude.
If our limit order was a sell order, the calculation would be `close * (1 + limit_delta)`.

#### Reversing

We've covered regular buy/sell limit orders that get matched whenever the price
is found to be the same or lower/higher than the requested price. Now, we can reverse the matching
logic with the argument `limit_reverse` to simulate buy/sell **stop** orders. For instance, for a buy 
stop order with an absolute delta of $100, it would search for a price that is $100 higher (as opposed 
to lower) than the current closing price and execute the limit order using that price:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     order_type="limit",
...     limit_delta=vbt.Param([0, 100, 5000, 10000]),
...     delta_format="absolute",
...     limit_reverse=True
... )
>>> pf.orders.price.to_pd()
limit_delta                   0        100      5000   10000
Open time                                                   
2021-02-18 00:00:00+00:00       NaN      NaN      NaN    NaN
2021-02-19 00:00:00+00:00  51552.61  51652.6      NaN    NaN
2021-02-20 00:00:00+00:00       NaN      NaN  56552.6    NaN
2021-02-21 00:00:00+00:00       NaN      NaN      NaN    NaN
2021-02-22 00:00:00+00:00       NaN      NaN      NaN    NaN
2021-02-23 00:00:00+00:00       NaN      NaN      NaN    NaN
2021-02-24 00:00:00+00:00       NaN      NaN      NaN    NaN
```

!!! info
    Why does the same delta of zero result in a different fill price without and with reversing? 
    Because upon hitting the second bar, the limit price is getting compared against the opening price.
    In the first example, the opening price is higher than the limit price, thus the limit order
    is executed using the limit price. In the second example, because the opening price is higher,
    the limit order is executed using the opening price.

#### Adjustment

In the theoretical section we discussed how it's possible to change any limit order from a callback.
There are three callbacks that we can use: a signal function (`signal_func_nb`), an adjustment function 
(`adjust_func_nb`), or a post-segment function (`adjust_func_nb`). The signal function is called at the 
beginning of a bar and should only be used if we need to define the four user-defined signals dynamically. 
The adjustment function is called at the beginning of the default signal function 
[dir_enex_signal_func_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.dir_enex_signal_func_nb) 
for direction-unaware and [ls_enex_signal_func_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.ls_enex_signal_func_nb) 
for direction-aware signals, which are provided via boolean arrays (statically). The post-segment 
function is called at the end of a bar for pre-calculating metrics such as portfolio returns. 

!!! note
    The adjustment function gets called only if the signals are provided statically,
    that is, `signal_func_nb` hasn't been overridden. Also note that overriding any of the
    functions above will disable caching of [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)!

Each of those functions accepts a named tuple with the current simulation context, which includes 
a one-dimensional record array [SignalContext.last_limit_info](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.last_limit_info) 
with the latest limit order information defined per column. By changing this array, we can change the 
limit order that is yet to be processed at this bar (or the next one for the post-segment function).
We will focus on the adjustment function since it's meant to work together with signal arrays.
This function should be Numba-compiled, take the current context variable (usually denoted with `c`)
and other custom arguments (`adjust_args`), and return nothing (`None`). Why nothing? Because
we can change all the relevant information in the context. Let's create a function
that cancels any pending limit order (that is, it hasn't been executed at the entry bar).
Limit orders will be created using the opening price and a parametrized custom delta:

```pycon
>>> @njit
... def adjust_func_nb(c):  # (1)!
...     limit_info = c.last_limit_info[c.col]  # (2)!
...     if limit_info.creation_idx != -1:  # (3)!
...         if c.i - limit_info.creation_idx >= 1:  # (4)!
...             vbt.pf_nb.clear_limit_info_nb(limit_info)  # (5)!

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price="open",
...     order_type="limit",
...     limit_delta=vbt.Param([0, 0.1]),
...     adjust_func_nb=adjust_func_nb
... )
>>> pf.orders.price.to_pd()
limit_delta                     0.0             0.1        
symbol                      BTCUSDT ETHUSDT BTCUSDT ETHUSDT
Open time                                                  
2021-02-18 00:00:00+00:00  52117.67  1849.7     NaN     NaN
2021-02-19 00:00:00+00:00       NaN     NaN     NaN     NaN
2021-02-20 00:00:00+00:00       NaN     NaN     NaN     NaN
2021-02-21 00:00:00+00:00       NaN     NaN     NaN     NaN
2021-02-22 00:00:00+00:00       NaN     NaN     NaN     NaN
2021-02-23 00:00:00+00:00       NaN     NaN     NaN     NaN
2021-02-24 00:00:00+00:00       NaN     NaN     NaN     NaN
```

1. Must take [SignalContext](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext) 
2. Get the pending limit order information for the current column (asset)
3. Check whether there is any limit order pending
4. Check whether the order has been pending for one bar or longer
5. Use [clear_limit_info_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.clear_limit_info_nb) to cancel the order

We can see that without a delta, each order could be filled already at the entry bar. With a delta of 10%
though, the orders couldn't be filled at the entry bar, and thus they were cancelled at the beginning
of the second bar (19th February). Let's do the opposite: if a limit order is still pending,
override its price with the valuation price and delta with 1% to execute at this bar:

```pycon
>>> @njit
... def adjust_func_nb(c):
...     limit_info = c.last_limit_info[c.col]
...     if limit_info.creation_idx != -1:
...         if c.i - limit_info.creation_idx >= 1:
...             limit_info.init_price = -np.inf  # (1)!
...             limit_info.delta = 0.01

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price="open",
...     order_type="limit",
...     limit_delta=vbt.Param([0.1, 0.2]),
...     adjust_func_nb=adjust_func_nb
... )
>>> pf.orders.price.to_pd()
limit_delta                       0.1                    0.2           
symbol                        BTCUSDT    ETHUSDT     BTCUSDT    ETHUSDT
Open time                                                              
2021-02-18 00:00:00+00:00         NaN        NaN         NaN        NaN
2021-02-19 00:00:00+00:00  51037.0839  1919.5209  51037.0839  1919.5209
2021-02-20 00:00:00+00:00         NaN        NaN         NaN        NaN
2021-02-21 00:00:00+00:00         NaN        NaN         NaN        NaN
2021-02-22 00:00:00+00:00         NaN        NaN         NaN        NaN
2021-02-23 00:00:00+00:00         NaN        NaN         NaN        NaN
2021-02-24 00:00:00+00:00         NaN        NaN         NaN        NaN
```

1. Negative infinity will be translated into the valuation price, that is, the latest
known price at the time of calling a callback, which is the opening price for adjustment
and signal callbacks, and the closing price for post-segment callbacks. Note that
you cannot use positive infinity for the closing price because it's unknown!

The same way as we cancelled an order, we can also create a new order without the need of 
user-defined signals - basically out of nothing!

```pycon
>>> @njit
... def adjust_func_nb(c, custom_delta):
...     limit_info = c.last_limit_info[c.col]
...     if c.i == 0:  # (1)!
...         curr_delta = vbt.pf_nb.select_nb(c, custom_delta)
...         vbt.pf_nb.set_limit_info_nb(limit_info, c.i, delta=curr_delta)  # (2)!

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     broadcast_named_args=dict(custom_delta=vbt.Param([0, 0.1])),  # (3)!
...     adjust_func_nb=adjust_func_nb,
...     adjust_args=(vbt.Rep("custom_delta"),)  # (4)!
... )
>>> pf.orders.price.to_pd()
custom_delta                    0.0                0.1         
symbol                      BTCUSDT ETHUSDT    BTCUSDT  ETHUSDT
Open time                                                      
2021-02-18 00:00:00+00:00  52117.67  1849.7        NaN      NaN
2021-02-19 00:00:00+00:00       NaN     NaN        NaN      NaN
2021-02-20 00:00:00+00:00       NaN     NaN        NaN      NaN
2021-02-21 00:00:00+00:00       NaN     NaN        NaN      NaN
2021-02-22 00:00:00+00:00       NaN     NaN        NaN  1664.73
2021-02-23 00:00:00+00:00       NaN     NaN  46905.903      NaN
2021-02-24 00:00:00+00:00       NaN     NaN        NaN      NaN
```

1. Create a signal order at the very first bar only
2. Use [set_limit_info_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.set_limit_info_nb).
Take a look at its signature to learn more about defaults. For example, it uses the current valuation
price, the infinity size, and both directions by default.
3. To make a custom argument parameterizable, we need to define it as a flexible argument 
in `broadcast_named_args`, that is, it must be able to be defined per element
4. Don't forget to put a comma after a single element in a tuple

We just created limit orders dynamically, without any signals :sparkles:

### Stop orders

Just like limit orders, stop orders are executed only once a price condition is met. Their information is 
also stored per column inside a one-dimensional record array. But in contrast to limit orders,
they are just deterred user-defined signals that result in a market or limit order to mainly close out 
an existing position. They are created upon a successful entry order that either opens, increases,
or reverses a position (i.e., closes out and then enters a new position in the opposite direction). 
There are three stop order types: SL (`sl_stop`), TSL and TTP (`tsl_stop` and `tsl_th`, acting as one order), 
and TP (`tp_stop`). Each one of them is stored and tracked separately, but they all act as a one-cancels-all 
(OCA) order where one order is triggered in full while the others are automatically canceled. Any
pending stop orders get also canceled if the entered position has been closed out by other means.

Let's place a stop loss order of 10%:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=0.1,
... )
>>> pf.orders.price.to_pd()
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  51552.60  1939.610
2021-02-19 00:00:00+00:00       NaN       NaN
2021-02-20 00:00:00+00:00       NaN       NaN
2021-02-21 00:00:00+00:00       NaN       NaN
2021-02-22 00:00:00+00:00       NaN  1745.649
2021-02-23 00:00:00+00:00  46397.34       NaN
2021-02-24 00:00:00+00:00       NaN       NaN
```

We can see that the entered long position has been exited at the price `51552.60 * 0.9 = 46397.34`.
To test multiple stop values, we can wrap it with the class [Param](/api/utils/params/#vectorbtpro.utils.params.Param):

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=vbt.Param([np.nan, 0.1, 0.2]),  # (1)!
... )
>>> pf.orders.price.to_pd()
sl_stop                        NaN                0.1                0.2  \\
symbol                     BTCUSDT  ETHUSDT   BTCUSDT   ETHUSDT  BTCUSDT   
Open time                                                                  
2021-02-18 00:00:00+00:00  51552.6  1939.61  51552.60  1939.610  51552.6   
2021-02-19 00:00:00+00:00      NaN      NaN       NaN       NaN      NaN   
2021-02-20 00:00:00+00:00      NaN      NaN       NaN       NaN      NaN   
2021-02-21 00:00:00+00:00      NaN      NaN       NaN       NaN      NaN   
2021-02-22 00:00:00+00:00      NaN      NaN       NaN  1745.649      NaN   
2021-02-23 00:00:00+00:00      NaN      NaN  46397.34       NaN      NaN   
2021-02-24 00:00:00+00:00      NaN      NaN       NaN       NaN      NaN   

sl_stop                              
symbol                      ETHUSDT  
Open time                            
2021-02-18 00:00:00+00:00  1939.610  
2021-02-19 00:00:00+00:00       NaN  
2021-02-20 00:00:00+00:00       NaN  
2021-02-21 00:00:00+00:00       NaN  
2021-02-22 00:00:00+00:00  1551.688  
2021-02-23 00:00:00+00:00       NaN  
2021-02-24 00:00:00+00:00       NaN  
```

1. NaN means no stop order

Since each stop argument and its accompanying information broadcast together with the data,
we can provide them as fully-fledged arrays. Below, we're defining a stop loss based on the ATR:

```pycon
>>> atr = data.run("atr").atr
>>> sub_atr = atr.loc["2021-02-18":"2021-02-24"]

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=sub_atr / sub_data.close
... )
>>> pf.orders.price.to_pd()
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2021-02-18 00:00:00+00:00  51552.600000  1939.610000
2021-02-19 00:00:00+00:00           NaN          NaN
2021-02-20 00:00:00+00:00           NaN  1805.283249
2021-02-21 00:00:00+00:00           NaN          NaN
2021-02-22 00:00:00+00:00  48335.716826          NaN
2021-02-23 00:00:00+00:00           NaN          NaN
2021-02-24 00:00:00+00:00           NaN          NaN
```

What about absolute stop values? They can be specified the same way as for limit orders:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data.select("BTCUSDT"),
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=100,
...     delta_format="absolute"
... )
>>> pf.orders.price.to_pd()
Open time
2021-02-18 00:00:00+00:00    51552.6
2021-02-19 00:00:00+00:00    51452.6
2021-02-20 00:00:00+00:00        NaN
2021-02-21 00:00:00+00:00        NaN
2021-02-22 00:00:00+00:00        NaN
2021-02-23 00:00:00+00:00        NaN
2021-02-24 00:00:00+00:00        NaN
Freq: D, dtype: float64
```

Since the simulator can keep track of multiple stop types at the same time, we can
simulate a specific risk-to-reward (R/R) ratio by setting a stop loss and a take profit.
For instance, let's simulate the risk-to-reward ratio of 1/2 based on the ATR:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=1 * sub_atr / sub_data.close,
...     tp_stop=2 * sub_atr / sub_data.close
... )
>>> pf.orders.price.to_pd()
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2021-02-18 00:00:00+00:00  51552.600000  1939.610000
2021-02-19 00:00:00+00:00           NaN          NaN
2021-02-20 00:00:00+00:00           NaN  1805.283249
2021-02-21 00:00:00+00:00  57986.366348          NaN
2021-02-22 00:00:00+00:00           NaN          NaN
2021-02-23 00:00:00+00:00           NaN          NaN
2021-02-24 00:00:00+00:00           NaN          NaN
```

The TP order won in the first column, the SL order won in the second column. But is there
a way to see the order type and preferably the date of the signal that initiated it? 
Sure, vectorbt can fulfill all of your wishes! :genie:

```pycon
>>> pf.orders.records_readable  # (1)!
   Order Id   Column              Signal Index            Creation Index  \\
0         0  BTCUSDT 2021-02-18 00:00:00+00:00 2021-02-18 00:00:00+00:00   
1         1  BTCUSDT 2021-02-18 00:00:00+00:00 2021-02-21 00:00:00+00:00   
2         0  ETHUSDT 2021-02-18 00:00:00+00:00 2021-02-18 00:00:00+00:00   
3         1  ETHUSDT 2021-02-18 00:00:00+00:00 2021-02-20 00:00:00+00:00   

                 Fill Index      Size         Price  Fees  Side    Type  \\
0 2021-02-18 00:00:00+00:00  0.001940  51552.600000   0.0   Buy  Market   
1 2021-02-21 00:00:00+00:00  0.001940  57986.366348   0.0  Sell  Market   
2 2021-02-18 00:00:00+00:00  0.051557   1939.610000   0.0   Buy  Market   
3 2021-02-20 00:00:00+00:00  0.051557   1805.283249   0.0  Sell  Market   

  Stop Type  
0      None  
1        TP  
2      None  
3        SL  

>>> pf.orders.stop_type.to_pd(mapping=True)  # (2)!
symbol                    BTCUSDT ETHUSDT
Open time                                
2021-02-18 00:00:00+00:00    None    None
2021-02-19 00:00:00+00:00    None    None
2021-02-20 00:00:00+00:00    None      SL
2021-02-21 00:00:00+00:00      TP    None
2021-02-22 00:00:00+00:00    None    None
2021-02-23 00:00:00+00:00    None    None
2021-02-24 00:00:00+00:00    None    None
```

1. Orders are represented as rows in a compressed record array. Specific information as a column.
2. Specific information as a time series

As we can read form the first DataFrame, the first order is a buy market order (`Side` and `Type`) 
that was filled on the 18th February (`Fill Index`). The second order in the same column is a sell 
market order resulting from a TP order (`Stop Type`) that was issued on the 18th February (`Signal Index`) 
but actually filled on the 21st February (`Fill Index`). The column `Creation Index` is the same
as the column `Fill Index` since all orders are market orders. The second DataFrame is a common
representation of a single piece of information in relation to the time and asset; it's best
used when the information should be analyzed as a time series with Pandas.

Finally, let's talk about multiple stop configurations. Here's how to test each stop type independently,
that is, SL of 10%, TSL of 10%, TTP of 10% with a 10%-threshold, and TP of 10%:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop= vbt.Param([0.1,    np.nan, np.nan, np.nan], product_idx=0),
...     tsl_stop=vbt.Param([np.nan, 0.1,    0.1,    np.nan], product_idx=0),
...     tsl_th=  vbt.Param([np.nan, np.nan, 0.1,    np.nan], product_idx=0),
...     tp_stop= vbt.Param([np.nan, np.nan, np.nan, 0.1   ], product_idx=0),
... )
>>> pf.total_return
sl_stop  tsl_stop  tsl_th  tp_stop  symbol 
0.1      NaN       NaN     NaN      BTCUSDT   -0.100000
                                    ETHUSDT   -0.100000
NaN      0.1       NaN     NaN      BTCUSDT    0.018717
                                    ETHUSDT   -0.052332
NaN      0.1       0.1     NaN      BTCUSDT    0.018717
                                    ETHUSDT   -0.163033
NaN      NaN       NaN     0.1      BTCUSDT    0.100000
                                    ETHUSDT   -0.163033
Name: total_return, dtype: float64
```

To build a product of various stop types, we can specify unique (non-repeating) values
and omit the `product_idx` argument:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=vbt.Param([np.nan, 0.1]),
...     tsl_stop=vbt.Param([np.nan, 0.1]),
...     tsl_th=vbt.Param([np.nan, 0.1]),
...     tp_stop=vbt.Param([np.nan, 0.1]),
... )
```

!!! warning
    The number of columns generated from a Cartesian product of multiple parameter combinations
    grows exponentially with the number of parameter combinations - show mercy to your RAM!

But this would include a combination where `tsl_th` is not NaN but `tsl_stop` is NaN, 
which doesn't make any sense. What we need though are three combinations: no TSL/TTP stop,
TSL stop, and TTP stop. Thus, we need to combine those two arguments manually and link them 
by the same `product_idx`:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=vbt.Param([np.nan, 0.1], product_idx=0),
...     tsl_stop=vbt.Param([np.nan, 0.1, 0.1], product_idx=1),
...     tsl_th=vbt.Param([np.nan, np.nan, 0.1], product_idx=1),
...     tp_stop=vbt.Param([np.nan, 0.1], product_idx=2),
... )
```

!!! note
    If some arguments require a product index, it must be defined for all arguments.

We can then call some metric and analyze it using Pandas. For example, let's calculate the average
return of using 1) either SL or TP, or 2) SL and TP together:

```pycon
>>> total_return = pf.total_return
>>> sl_stops = total_return.index.get_level_values("sl_stop")
>>> tp_stops = total_return.index.get_level_values("tp_stop")

>>> total_return[np.isnan(sl_stops) | np.isnan(tp_stops)].mean()
-0.045462468872515135

>>> total_return[~np.isnan(sl_stops) & ~np.isnan(tp_stops)].mean()
0.0
```

Seems like combining SL and TP works better on our "huge" time series with 7 points :upside_down_face:

The examples above work only if we're testing arguments provided as scalars (single values). How can
we backtest various combinations of stop values provided as arrays though, such as those computed from 
the ATR? Performing that manually would be highly difficult because we would need to tile each stop 
type array by the number of stop values and then combine all stop type arrays using the Cartesian product.
An easier approach is to build a basic indicator that does the preparation of arrays for us:

```pycon
>>> StopOrderPrep = vbt.IF.from_expr("""
...     sl_stop = @p_sl_mult * @in_atr / @in_close
...     tp_stop = @p_tp_mult * @in_atr / @in_close
...     sl_stop, tp_stop
... """)
>>> stop_order_prep = StopOrderPrep.run(
...     close=sub_data.close, 
...     atr=sub_atr,
...     sl_mult=[np.nan, 1],  # (1)!
...     tp_mult=[np.nan, 1],
...     param_product=True
... )
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     sl_stop=stop_order_prep.sl_stop,
...     tp_stop=stop_order_prep.tp_stop,
... )
>>> pf.total_return
custom_sl_mult  custom_tp_mult  symbol 
NaN             NaN             BTCUSDT   -0.036398
                                ETHUSDT   -0.163033
NaN             1.0             BTCUSDT    0.062400
                                ETHUSDT   -0.163033
1.0             NaN             BTCUSDT   -0.062400
                                ETHUSDT   -0.069255
1.0             1.0             BTCUSDT    0.062400
                                ETHUSDT   -0.069255
Name: total_return, dtype: float64
```

1. NaN means no order, 1 means one ATR divided by the close

#### Entry point

The entry price of stop orders can be configured using the argument `stop_entry_price`:
it can take either the price itself, or any option from the enumerated type 
[StopEntryPrice](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopEntryPrice). 
To be able to distinguish between real prices and options, the options have a negative sign. By default, 
the entry price is the closing price, even if the entry order itself was filled before the 
closing price, such as at the opening price; this is because we cannot fill a stop order at 
the same bar as the entry order, that's why we're postponing the first check to the next bar.
Let's see what happens if we define a stop order that can be executed already at the first bar:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price="open",
...     tp_stop=0.005,
...     stop_entry_price="fillprice"
... )
>>> pf.orders.price.to_pd()
symbol                         BTCUSDT  ETHUSDT
Open time                                      
2021-02-18 00:00:00+00:00  52117.67000  1849.70
2021-02-19 00:00:00+00:00  52378.25835  1938.91
2021-02-20 00:00:00+00:00          NaN      NaN
2021-02-21 00:00:00+00:00          NaN      NaN
2021-02-22 00:00:00+00:00          NaN      NaN
2021-02-23 00:00:00+00:00          NaN      NaN
2021-02-24 00:00:00+00:00          NaN      NaN
```

As we can see, even though the target price `52378.26` could theoretically be filled at the first 
bar as it's lower than the highest price of `52530.00` of that bar, the check was still postponed to 
the next bar because of the FS's limitation of one order per bar and asset.

#### Exit point

The number of options to define a stop **exit** price is relatively small: the stop price
and the closing price. If the stop has been matched at the beginning of a bar,
the stop price will become the opening price. The same for the closing price. By default, 
a stop order is executed using the stop price, but if the entire simulation should be done
on the closing prices only, the second option might be more appropriate.

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     tp_stop=0.05,
...     stop_exit_price=vbt.Param(["stop", "close"])
... )
>>> pf.orders.price.to_pd()
stop_exit_price                stop               close         
symbol                      BTCUSDT    ETHUSDT  BTCUSDT  ETHUSDT
Open time                                                       
2021-02-18 00:00:00+00:00  51552.60  1939.6100  51552.6  1939.61
2021-02-19 00:00:00+00:00  54130.23        NaN  55906.0      NaN
2021-02-20 00:00:00+00:00       NaN  2036.5905      NaN  1913.00
2021-02-21 00:00:00+00:00       NaN        NaN      NaN      NaN
2021-02-22 00:00:00+00:00       NaN        NaN      NaN      NaN
2021-02-23 00:00:00+00:00       NaN        NaN      NaN      NaN
2021-02-24 00:00:00+00:00       NaN        NaN      NaN      NaN
```

This way, we're effectively delaying the execution of a stop order until the end of a bar. Not only
we can change the default stop exit price, but also the default stop exit behavior:
using the argument `stop_exit_type` of the type [StopExitType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.StopExitType)
we can reduce/reverse/reverse+reduce the position instead of closing it. By default,
the position is being closed out by issuing an exit signal and disabling accumulation, but we can also
reverse it by issuing an opposite entry signal and/or reduce it by keeping the default accumulation:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     tp_stop=0.01,  # (1)!
...     stop_exit_type=vbt.Param(["close", "reverse"])
... )
>>> pf.orders.price.to_pd()
stop_exit_type                 close                reverse           
symbol                       BTCUSDT    ETHUSDT     BTCUSDT    ETHUSDT
Open time                                                             
2021-02-18 00:00:00+00:00  51552.600  1939.6100  51552.6000  1939.6100
2021-02-19 00:00:00+00:00  52068.126  1959.0061  52068.1260  1959.0061
2021-02-20 00:00:00+00:00        NaN        NaN  55346.9400  1935.4500
2021-02-21 00:00:00+00:00        NaN        NaN  56399.6019  1932.1300
2021-02-22 00:00:00+00:00        NaN        NaN  56834.4843  1914.1947
2021-02-23 00:00:00+00:00        NaN        NaN         NaN        NaN
2021-02-24 00:00:00+00:00        NaN        NaN         NaN        NaN
```

1. TP of 1%

Take a look at both columns under the parameter value `reverse`: they now have five
orders instead of two, how so? By reversing, we're closing the existing position and opening a new one 
but in the opposite direction, using a single order. Opening a new position will create a new stop 
order, thus we've created a never-ending cycle of conditional reversals :grimacing: If such a
cycle is unwanted, we can define `stop_exit_type` as an array where only a particular stop
is closed or reversed. Each value in this array should be located at the entry point rather
than exit point because this information (along with `stop_exit_price`) is order-anchored and not 
date-anchored! For example, let's reverse the first order only:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     tp_stop=0.01,
...     stop_exit_type=vbt.index_dict({0: "reverse", "_def": "close"})
... )
>>> pf.orders.price.to_pd()
symbol                       BTCUSDT    ETHUSDT
Open time                                      
2021-02-18 00:00:00+00:00  51552.600  1939.6100
2021-02-19 00:00:00+00:00  52068.126  1959.0061
2021-02-20 00:00:00+00:00  55346.940  1935.4500
2021-02-21 00:00:00+00:00        NaN        NaN
2021-02-22 00:00:00+00:00        NaN        NaN
2021-02-23 00:00:00+00:00        NaN        NaN
2021-02-24 00:00:00+00:00        NaN        NaN
```

The first position was reversed while the second position only exited.

What about order type? Upon exit, the order type becomes either a market or limit order.
This behavior is controlled by the argument `stop_order_type` of the type 
[OrderType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderType). There is also
an argument `stop_limit_delta` to specify a delta for the resulting limit order!
Let's test a 5% TP order as a stop market order vs. stop limit order with a delta of 1%:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     tp_stop=0.05,
...     stop_order_type=vbt.Param(["market", "limit"]),
...     stop_limit_delta=0.01
... )
>>> pf.orders.price.to_pd()
stop_order_type              market                  limit         
symbol                      BTCUSDT    ETHUSDT     BTCUSDT  ETHUSDT
Open time                                                          
2021-02-18 00:00:00+00:00  51552.60  1939.6100  51552.6000  1939.61
2021-02-19 00:00:00+00:00  54130.23        NaN  54671.5323      NaN
2021-02-20 00:00:00+00:00       NaN  2036.5905         NaN      NaN
2021-02-21 00:00:00+00:00       NaN        NaN         NaN      NaN
2021-02-22 00:00:00+00:00       NaN        NaN         NaN      NaN
2021-02-23 00:00:00+00:00       NaN        NaN         NaN      NaN
2021-02-24 00:00:00+00:00       NaN        NaN         NaN      NaN
```

The latest stop limit order couldn't be filled because the highest price in the entire column 
`ETHUSDT` is `2042.34` while the requested limit price is `1939.61 * 1.05 * 1.01 = 2056.96`.
But even if there was a price higher on the 20th February: unless the stop price was triggered
at open, the simulator wouldn't be able to use the entire candle to match the limit price
because there is no intra-bar data to prove that the limit price can be hit strictly
after the stop price. The only information the simulator could have used in this case is the 
closing price: only if it was higher than the limit price, there would be a guarantee that
the limit price has actually been hit.

#### Conflicts

Conflicts between stops orders and user-defined signals enjoy the same resolution logic
as limit orders, the only difference is in the argument naming: `upon_adj_stop_conflict` and 
`upon_opp_stop_conflict`. By default, whenever a user-defined signal in any direction is encountered,
nothing happens: any stop order remains pending and the signal gets executed as usual; that's because 
pending stop orders are automatically cancelled whenever the position gets closed out. Let's cancel 
any signal when there is a stop order pending:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, X, O, X, O, X]),
...     exits=  pd.Series([O, X, O, X, O, X, O]),
...     direction="both",
...     sl_stop=0.1,
...     upon_adj_stop_conflict="keepignore",
...     upon_opp_stop_conflict="keepignore",
... )
>>> pf.orders.price.to_pd()
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  51552.60  1939.610
2021-02-19 00:00:00+00:00       NaN       NaN
2021-02-20 00:00:00+00:00       NaN       NaN
2021-02-21 00:00:00+00:00       NaN       NaN
2021-02-22 00:00:00+00:00       NaN  1745.649
2021-02-23 00:00:00+00:00  46397.34  1577.890
2021-02-24 00:00:00+00:00  49676.20       NaN
```

#### Adjustment

Similarly to limit orders, stop orders are also stored in one-dimensional record arrays 
where information is laid out per column: 
[SignalContext.last_sl_info](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.last_sl_info) 
for SL, [SignalContext.last_tsl_info](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.last_tsl_info) 
for TSL and TTP, and [SignalContext.last_tp_info](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext.last_tp_info) for TP.
For instance, create an adjustment function that creates one TP and one SL order based on a maximum 
allowed, absolute, parameterizable profit and loss (PnL):

```pycon
>>> @njit
... def adjust_func_nb(c, max_loss, max_profit):
...     position_now = c.last_position[c.col]  # (1)!
...     if position_now != 0:  # (2)!
...         sl_info = c.last_sl_info[c.col]
...         tp_info = c.last_tp_info[c.col]
...         ml = vbt.pf_nb.select_nb(c, max_loss)  # (3)!
...         mp = vbt.pf_nb.select_nb(c, max_profit)
...
...         if not vbt.pf_nb.is_stop_active_nb(sl_info):  # (4)!
...             sl_info.stop = ml / (sl_info.init_price * position_now)  # (5)!
...         if not vbt.pf_nb.is_stop_active_nb(tp_info):
...             tp_info.stop = mp / (sl_info.init_price * position_now)

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     adjust_func_nb=adjust_func_nb,
...     adjust_args=(vbt.Rep("max_loss"), vbt.Rep("max_profit")),
...     broadcast_named_args=dict(
...         max_loss=10,
...         max_profit=10
...     )
... )
>>> pf.exit_trades.pnl.to_pd()  # (6)!
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00      NaN      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00     10.0      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00      NaN    -10.0
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Get the size of the current position. Remember that fields that are not associated
with cash are accessible via column!
2. We're in the market when the position is not zero
3. Our parameters broadcast against the data, thus we need to select the current
element flexibly
4. There is a handy function [is_stop_active_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.is_stop_active_nb)
to check whether the current stop order is active (i.e., the stop value is not NaN)
5. Divide the maximum loss/profit by the notional value to get the optimal stop loss/take profit
6. Get PnL of each trade as a time series

!!! hint
    To avoid re-compiling the entire simulation function with each new run, define the 
    adjustment function in a different cell or even file. Caching won't help though
    since Numba doesn't allow for caching callbacks.

Our losses and profits are now capped by $10! Here's what happened. Whenever we provide a custom
adjustment function or any other callback, the simulator switches to the flexible mode and enables 
stop orders. Then, whenever a new position is opened or an existing one is increased, the simulator
initializes each stop type with the provided arguments. But since we haven't passed anything
related to SL and TP, their default stop value is NaN, thus they are inactive by default. But regardless
of that, they still contain valuable information, such as the initial price. Hence, what's left for us 
is just to check whether the respective stop is inactive and override the stop value to activate it. 
We could have also used the functions [set_sl_info_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.set_sl_info_nb)
and [set_tp_info_nb](/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.set_tp_info_nb)
to set the entire information, but there was no need for that.

### Dynamic

We've learned that order-related arguments such as `size` act as a facade of the backtesting process 
with FS: they are meant to be anchored by time rather than signals, so we cannot change
them dynamically. But there is a magic trick: if we define any argument as one-dimensional
array with just one single element per column and change this element in a callback before the signals
for this bar and asset are executed, the simulator will recognize its shape and pick that one element
using flexible indexing! The only requirement is that the array must be writeable. 
Let's demonstrate this behavior by adding 10$ to the order size of each new order:

```pycon
>>> @njit
... def signal_func_nb(c, entries, exits, size):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     if is_entry:
...         size[c.col] += 10
...     return is_entry, is_exit, False, False

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("entries"), 
...         vbt.Rep("exits"), 
...         vbt.Rep("size")  # (1)!
...     ),
...     size=vbt.RepEval("np.full(wrapper.shape_2d[1], 0.0)"),  # (2)!
...     size_type="value",
...     accumulate=True,
...     broadcast_named_args=dict(entries=True, exits=False)  # (3)!
... )
>>> pf.orders.value.to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00     10.0     10.0
2021-02-19 00:00:00+00:00     20.0     20.0
2021-02-20 00:00:00+00:00     30.0     30.0
2021-02-21 00:00:00+00:00     40.0     40.0
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Forward the size array as well
2. Use an expression evaluation template to delay the creation of the array until the target shape is known.
Here, we create a one-dimensional array with values by column.
3. Create a buy order at each bar and asset

Our algorithm added 10$ to each order in a column until the column has run out of money.
To demonstrate the full power of this approach though, let's create a take profit ladder!
We'll create and keep track of a ladder of two take profit stops: one of 5% that will sell 50% of 
the initial position, and one of 10% that will sell the remainder (100%) of the position.
Also, we'll show how to group arguments using named tuples and define custom record arrays
that hold complex temporary information:

```pycon
>>> from collections import namedtuple

>>> Signals = namedtuple("Signals", [  # (1)!
...     "entries", 
...     "exits"
... ])
>>> Orders = namedtuple("Orders", [  # (2)!
...     "size", 
...     "accumulate"
... ])
>>> Ladders = namedtuple("Ladders", [  # (3)!
...     "tp1", 
...     "tp2", 
...     "tp1_pct", 
...     "tp2_pct"
... ])

>>> ladder_info_dt = np.dtype([  # (4)!
...     ("init_idx", np.int_),
...     ("init_price", np.float_),
...     ("tp1", np.float_),
...     ("tp2", np.float_),
...     ("tp1_pct", np.float_),
...     ("tp2_pct", np.float_)
... ], align=True)

>>> @njit  # (5)!
... def signal_func_nb(c, signals, orders, ladders, last_ladder_info):  # (6)!
...     is_entry = vbt.pf_nb.select_nb(c, signals.entries)  # (7)!
...     is_exit = vbt.pf_nb.select_nb(c, signals.exits)
...     position_now = c.last_position[c.col]
...     if position_now == 0 and is_entry:
...         orders.size[c.col] = np.inf  # (8)!
...         orders.accumulate[c.col] = False
...         return True, False, False, False
...     if position_now > 0 and is_exit:
...         orders.size[c.col] = np.inf
...         orders.accumulate[c.col] = False
...         return False, True, False, False
...     if position_now == 0:
...         return False, False, False, False
...
...     tp_info = c.last_tp_info[c.col]
...     ladder_info = last_ladder_info[c.col]  # (9)!
...     last_order = c.order_records[c.order_counts[c.col] - 1, c.col]  # (10)!
...     if not vbt.pf_nb.is_stop_active_nb(tp_info):  # (11)!
...         if last_order.stop_type == -1:  # (12)!
...             ladder_info.init_idx = tp_info.init_idx  # (13)!
...             ladder_info.init_price = tp_info.init_price
...             ladder_info.tp1 = vbt.pf_nb.select_nb(c, ladders.tp1)
...             ladder_info.tp2 = vbt.pf_nb.select_nb(c, ladders.tp2)
...             ladder_info.tp1_pct = vbt.pf_nb.select_nb(c, ladders.tp1_pct)
...             ladder_info.tp2_pct = vbt.pf_nb.select_nb(c, ladders.tp2_pct)
...             tp_info.stop = ladder_info.tp1  # (14)!
...             tp_info.exit_type = vbt.pf_enums.StopExitType.CloseReduce
...             orders.size[c.col] = ladder_info.tp1_pct * position_now
...             orders.accumulate[c.col] = True
...         else:  # (15)!
...             vbt.pf_nb.set_tp_info_nb(  # (16)!
...                 tp_info, 
...                 ladder_info.init_idx,  # (17)!
...                 init_price=ladder_info.init_price,
...                 stop=ladder_info.tp2,
...                 exit_type=vbt.pf_enums.StopExitType.CloseReduce
...             )
...             orders.size[c.col] = ladder_info.tp2_pct * position_now  # (18)!
...             orders.accumulate[c.col] = True
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(  # (19)!
...         vbt.RepEval(
...             "Signals(entries, exits)",
...             context=dict(Signals=Signals)  # (20)!
...         ),
...         vbt.RepEval(
...             "Orders(size, accumulate)",
...             context=dict(Orders=Orders)
...         ),
...         vbt.RepEval(
...             "Ladders(tp1, tp2, tp1_pct, tp2_pct)",
...             context=dict(Ladders=Ladders)
...         ),
...         vbt.RepEval(
...             "np.empty(wrapper.shape_2d[1], dtype=ladder_info_dt)", 
...             context=dict(ladder_info_dt=ladder_info_dt)
...         )
...     ),
...     size=vbt.RepEval("np.full(wrapper.shape_2d[1], np.inf)"),
...     accumulate=vbt.RepEval("np.full(wrapper.shape_2d[1], False)"),
...     broadcast_named_args=dict(
...         entries=pd.Series([X, O, O, O, O, O, O]), 
...         exits=  pd.Series([O, O, O, O, O, O, O]),
...         tp1=0.05,
...         tp2=0.1,
...         tp1_pct=0.5,
...         tp2_pct=1.0
...     )
... )
>>> pf.asset_flow
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.051557
2021-02-19 00:00:00+00:00 -0.00097  0.000000
2021-02-20 00:00:00+00:00 -0.00097 -0.025778
2021-02-21 00:00:00+00:00  0.00000  0.000000
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000  0.000000
2021-02-24 00:00:00+00:00  0.00000  0.000000
```

1. Create a [namedtuple](https://realpython.com/python-namedtuple/) to define a container
for multiple arrays. Otherwise, the amount of arrays passed to the signal function would take
a lot of space and be poorly readable. This named tuple contains two signal arrays: entries and exits.
2. The second named tuple contains order-related arrays that are meant to be
overridden in the signal function: size and accumulation. Why them? Because they should differ
for regular and stop orders.
3. Since providing information on ladders requires creation of at least 4 arrays, let's put them into
a named tuple too!
4. Latest ladder information for each column should be stored inside a record array with a 
complex data type. Imagine each field being a column in a DataFrame. We could have also used a 
named tuple with multiple arrays, but this format is more appropriate for this task.
5. If you're finished with the function, don't forget to put it into another cell to avoid re-compilation!
Also, don't cache it because it might break if you change some named tuples.
6. Our function takes the context, three named tuples, and one temporary array
7. First, we need to take care of any user-defined signal. Here, signals enjoy a higher priority
and are executed ahead of stop orders.
8. We want the signal to be executed with a different size and accumulation
than stop orders, thus override the order information
9. Similarly to built-in information arrays, our latest ladder information is also laid out per column
10. Filled order records are stored in a two-dimensional array. We can query the latest filled order by
using the number of records that have been filled up to this point in each column.
11. If the current stop order is not active, but we're in the market, we need to activate it
12. If the latest order is a user-defined order (i.e., not a stop order), then activate
the first TP order
13. Collect all the relevant information and put it into the temporary array
14. Activate the first TP order. We also need to define a custom size and turn on accumulation
if we want the stop order to reduce the position instead of closing it completely.
15. If the latest order is a stop order, then apparently the first take profit was hit.
Thus, activate the second TP order.
16. After a (previous) stop order was hit, it's information was deleted completely, hence it's not enough
to only set the stop value - we need to set all information
17. Since our stop values are defined relative to the entry price of the entire position, 
re-use the information that we stored previously
18. Size percentages are relative to the current position, not to the initial one
19. Use templates to construct named tuples and record arrays. The arrays defined in
`broadcast_named_args` will be automatically recognized in template expressions.
20. We need to provide the classes via a context since they are not automatically available in the expressions

The algorithm reduced the position in the first column by 50% after hitting the first stop, and 
then by another 100% after hitting the second stop. In the second column, only the first stop 
was hit. The code above is a great template for defining flexible strategies in the most convenient way,
and not only that: we can easily backtest arbitrary ladder configurations thanks to broadcasting
and flexible indexing as each of our custom arguments such as `tp1` behaves the same way as 
the built-in arguments in [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals).

## Grouping

Until now, we processed the columns `BTCUSDT` and `ETHUSDT` separately, that is, the second
column is processed strictly after the first column. But what if there is a need to put both
columns into the same portfolio for analysis? By introducing a grouping, we can make a group
of columns to be treated as a single column (read more [here](/documentation/portfolio/from-orders/#grouping_1)).
Here, we need to distinguish between grouping after and before the simulation.

### After simulation

When the main scenario is to simulate columns __separately__ and then subsequently analyze them as a whole,
we can group-by the columns after the simulation, either when querying a specific metric of interest:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=      pd.Series([X, O, O, O, O, O, O]),
...     exits=        pd.Series([O, O, O, X, O, O, O]),
...     short_entries=pd.Series([O, O, O, O, X, O, O]),
...     short_exits=  pd.Series([O, O, O, O, O, O, X]),
... )
>>> pf.get_value(group_by=True)
Open time
2021-02-18 00:00:00+00:00    200.000000
2021-02-19 00:00:00+00:00    209.238037
2021-02-20 00:00:00+00:00    206.946937
2021-02-21 00:00:00+00:00    211.045749
2021-02-22 00:00:00+00:00    211.045749
2021-02-23 00:00:00+00:00    232.943589
2021-02-24 00:00:00+00:00    228.775332
Freq: D, Name: group, dtype: float64
```

Or, by quick-fixing the entire portfolio:

```pycon
>>> grouped_pf = pf.replace(wrapper=pf.wrapper.replace(group_by=True))  # (1)!
>>> grouped_pf.value
Open time
2021-02-18 00:00:00+00:00    200.000000
2021-02-19 00:00:00+00:00    209.238037
2021-02-20 00:00:00+00:00    206.946937
2021-02-21 00:00:00+00:00    211.045749
2021-02-22 00:00:00+00:00    211.045749
2021-02-23 00:00:00+00:00    232.943589
2021-02-24 00:00:00+00:00    228.775332
Freq: D, Name: group, dtype: float64
```

1. Group-by instruction is part of the portfolio's wrapper

What the portfolio above did was aggregate various metrics such as the initial capital and time series 
such as the cash flows along the column axis. There are some time series that cannot be aggregated
though, such as the asset flows:

```pycon
>>> grouped_pf.asset_flow
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.001940  0.051557
2021-02-19 00:00:00+00:00  0.000000  0.000000
2021-02-20 00:00:00+00:00  0.000000  0.000000
2021-02-21 00:00:00+00:00 -0.001940 -0.051557
2021-02-22 00:00:00+00:00 -0.002059 -0.056080
2021-02-23 00:00:00+00:00  0.000000  0.000000
2021-02-24 00:00:00+00:00  0.002059  0.056080
```

We can still disable the grouping for individual metrics by passing `group_by=False`. What we
cannot do after the simulation though is to introduce cash sharing because it can be only applied 
during the simulation where it has real consequences on the trading logic.

### Before simulation

Adding a group-by instruction before the simulation may or may not have consequences on the simulation.
For instance, grouping with cash sharing will always have consequences, while grouping without cash sharing
will only have consequences when FS is run in the flexible mode (that is, when any of the default callbacks 
are overridden); in this case, columns are grouped for the user to take advantage of the grouping in
the callbacks, while all the calculations are still done on the per-column basis as if there was no 
grouping at all. Let's make use of this fact and limit the number of active positions to just one in each group:

```pycon
>>> @njit
... def signal_func_nb(c, entries, exits):
...     is_entry = vbt.pf_nb.select_nb(c, entries)
...     is_exit = vbt.pf_nb.select_nb(c, exits)
...     other_in_position = False
...     for col in range(c.from_col, c.to_col):  # (1)!
...         if col != c.col and c.last_position[col] != 0:
...             other_in_position = True
...             break
...     if other_in_position:  # (2)!
...         return False, False, False, False
...     return is_entry, is_exit, False, False  # (3)!

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(vbt.Rep("entries"), vbt.Rep("exits")),
...     broadcast_named_args=dict(
...         entries=pd.DataFrame({
...             0: [X, O, O, O, O, O, X], 
...             1: [O, X, O, X, O, O, O]
...         }), 
...         exits=pd.DataFrame({
...             0: [O, O, X, O, X, O, O], 
...             1: [O, O, O, O, O, X, O]
...         })
...     ),
...     group_by=True  # (4)!
... )
>>> pf.asset_flow
symbol                     BTCUSDT   ETHUSDT
Open time                                   
2021-02-18 00:00:00+00:00  0.00194  0.000000
2021-02-19 00:00:00+00:00  0.00000  0.000000
2021-02-20 00:00:00+00:00 -0.00194  0.000000
2021-02-21 00:00:00+00:00  0.00000  0.051719
2021-02-22 00:00:00+00:00  0.00000  0.000000
2021-02-23 00:00:00+00:00  0.00000 -0.051719
2021-02-24 00:00:00+00:00  0.00218  0.000000
```

1. Go through the columns in this group and check whether there are open positions apart from this one
2. If true, return no signal
3. If false, return the signal
4. Put all columns into the same group. For multiple groups, use `vbt.ExceptLevel("symbol")`

The first signal in the column `ETHUSDT` couldn't execute because there was already a position
in the column `BTCUSDT`, but once it was closed, the second signal could finally go through.

#### Sorting

Orders are sorted only in groups with cash sharing, and only in two cases: when orders in 
different columns should be executed in different bar zones (i.e., at different times) such
that they need to be sorted by time, and when orders are instructed to be sorted by order value as 
part of the automatic call sequence enabled by `call_seq="auto"`. Both cases cannot be combined!
Let's take a look at the first case where the first column is executed using the closing price
and the second column using the opening price:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price=["close", "open"],  # (1)!
...     size=[100, 50],
...     size_type="value",
...     group_by=True,
...     cash_sharing=True  # (2)!
... )
>>> pf.orders.get_value(group_by=False).to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00     50.0     50.0
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Regular list will be transformed into a one-dimensional array, which will broadcast
against columns if the input data is two-dimensional
2. Turn on cash sharing

As we can see, the second column was executed first and has ended up using the half of the capital.
If both assets were executed in the same bar zone though, they wouldn't be sorted and
would be just normally executed from the left column to the right:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price="close",
...     size=[100, 50],
...     size_type="value",
...     group_by=True,
...     cash_sharing=True
... )
>>> pf.orders.get_value(group_by=False).to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00    100.0      NaN
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

Even though the second column requested $50 of assets, the simulator couldn't fulfill that request
because the default initial capital is $100, which has been all used up by the first column.
To sort by order value, use the automatic call sequence:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price="close",
...     size=[100, 50],
...     size_type="value",
...     group_by=True,
...     cash_sharing=True,
...     call_seq="auto"  # (1)!
... )
>>> pf.orders.get_value(group_by=False).to_pd()
symbol                     BTCUSDT  ETHUSDT
Open time                                  
2021-02-18 00:00:00+00:00     50.0     50.0
2021-02-19 00:00:00+00:00      NaN      NaN
2021-02-20 00:00:00+00:00      NaN      NaN
2021-02-21 00:00:00+00:00      NaN      NaN
2021-02-22 00:00:00+00:00      NaN      NaN
2021-02-23 00:00:00+00:00      NaN      NaN
2021-02-24 00:00:00+00:00      NaN      NaN
```

1. Here

This has made the second column execute first. But if we attempted to sort by order value when 
the columns had to execute their orders at different times, the simulation would fail:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price=["close", "open"],
...     size=[100, 50],
...     size_type="value",
...     group_by=True,
...     cash_sharing=True,
...     call_seq="auto"
... )
ValueError: Cannot sort orders by value if they are executed at different times
```

And it makes sense if we think more about it: if the simulator sorted both columns by value,
the second column would come first, but how it's possible if it should execute at the beginning of the bar?
Remember that each bar is divided into three zones: open, middle, and close. The only possibility
to sort orders by value is to use either the opening or the closing point of the bar. We cannot use
the middle of the bar because we don't really know what happens there. Since we give vectorbt
clues on where in the bar an order should execute by using `-np.inf` (bar open) and `np.inf` (bar close) 
as order price, passing any custom prices would be considered happening in the middle of a bar and 
thus fail as well:

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=pd.Series([X, O, O, O, O, O, O]),
...     price=sub_data.close,
...     size=[100, 50],
...     size_type="value",
...     group_by=True,
...     cash_sharing=True,
...     call_seq="auto"
... )
ValueError: Cannot sort orders by value if they are executed at different times
```

Even if we know that the provided array represents the closing price, vectorbt doesn't know about it.
Thus, make sure to use exclusively any option from [PriceType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceType)
when the automatic call sequence should be enabled. If you're using stop orders, make sure to
use `PriceType.Close` and `StopExitPrice.Close` to execute all orders at the closing price, for instance,
for rebalancing. Limit orders, on the other hand, are never executed at the closing price,
thus using them together with the automatic call sequence is hardly possible.

## Custom outputs

When the portfolio has been simulated, and we've got our new shiny portfolio instance, the first thing
we usually do is run various statistics. Since most statistics are based on returns, the portfolio
instance may sometimes need to repeatedly re-calculate the returns, which is a slow process because 
it has to translate order records into asset flows, cash flows, assets, cash, asset value, portfolio value,
and finally, returns. Such operations are known as "reconstructions" because they aim to 
derive a certain property of the simulation post factum. Hence, performance analysis is usually the 
main performance bottleneck in a typical backtesting pipeline. Gladly, there is an argument
`fill_returns` that, when enabled, pre-calculates the returns during the simulation and makes them 
available for all the metrics that need them (read more [here](/documentation/portfolio/from-orders/#filling-returns)):

```pycon
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=      pd.Series([X, O, O, O, O, O, O]),
...     exits=        pd.Series([O, O, O, X, O, O, O]),
...     short_entries=pd.Series([O, O, O, O, X, O, O]),
...     short_exits=  pd.Series([O, O, O, O, O, O, X]),
...     fill_returns=True
... )
>>> pf.returns
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000000  0.000000
2021-02-19 00:00:00+00:00  0.084446  0.007935
2021-02-20 00:00:00+00:00 -0.001159 -0.021483
2021-02-21 00:00:00+00:00  0.028069  0.010732
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00  0.096079  0.112338
2021-02-24 00:00:00+00:00 -0.013245 -0.023012
```

To check whether they make sense, compare them to the reconstructed returns:

```pycon
>>> pf.get_returns()  # (1)!
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000000  0.000000
2021-02-19 00:00:00+00:00  0.084446  0.007935
2021-02-20 00:00:00+00:00 -0.001159 -0.021483
2021-02-21 00:00:00+00:00  0.028069  0.010732
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00  0.096079  0.112338
2021-02-24 00:00:00+00:00 -0.013245 -0.023012
```

1. Every attribute with `get_` triggers a reconstruction

Under the hood, FS creates a named tuple with an uninitialized returns array and gradually
fills it at the end of each bar. Once the simulation is over, this tuple gets attached to
the created portfolio instance under the attribute 
[Portfolio.in_outputs](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.in_outputs):

```pycon
>>> pf.in_outputs
FSInOutputs(returns=array([[ 0.        ,  0.        ],
                           [ 0.08444579,  0.00793458],
                           [-0.00115927, -0.02148338],
                           [ 0.02806853,  0.01073183],
                           [ 0.        ,  0.        ],
                           [ 0.09607864,  0.11233812],
                           [-0.01324464, -0.02301153]]))
```

Since most things returned by the simulator are in NumPy format, we can use the method
[Portfolio.get_in_output](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.get_in_output)
to wrap any array with Pandas:

```pycon
>>> pf.get_in_output("returns")
symbol                      BTCUSDT   ETHUSDT
Open time                                    
2021-02-18 00:00:00+00:00  0.000000  0.000000
2021-02-19 00:00:00+00:00  0.084446  0.007935
2021-02-20 00:00:00+00:00 -0.001159 -0.021483
2021-02-21 00:00:00+00:00  0.028069  0.010732
2021-02-22 00:00:00+00:00  0.000000  0.000000
2021-02-23 00:00:00+00:00  0.096079  0.112338
2021-02-24 00:00:00+00:00 -0.013245 -0.023012
```

But there is one catch: the argument `fill_returns` is only available in the fixed simulation mode,
that is, when none of the default callbacks have been overridden. As soon as we enter the flexible mode,
we need to define our own in-output tuple under the argument `in_outputs` and fill it with any information 
we want, usually in the segment post-processing function `post_segment_func`. This callback takes the context 
[SignalSegmentContext](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalSegmentContext),
which is the same as [SignalContext](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalContext)
but without the information on the current column since a segment encompasses multiple columns
in a particular group at a particular row, hence, we need to go over the columns in the current 
segment manually. 

For demonstration, let's develop a post-segment callback that fills the portfolio returns the same way 
as done by the fixed FS mode! In addition, we will store the total portfolio return per column to avoid 
reconstructing it during the post-analysis phase:

```pycon
>>> @njit
... def post_segment_func_nb(c):
...     returns = c.in_outputs.returns  # (1)!
...     total_return = c.in_outputs.total_return
...     i = c.i
...     g = c.group
...     if c.cash_sharing:  # (2)!
...         returns[i, g] = c.last_return[g]
...         total_return[g] = c.last_value[g] / c.init_cash[g] - 1
...     else:  # (3)!
...         for col in range(c.from_col, c.to_col):  # (4)!
...             returns[i, col] = c.last_return[col]
...             total_return[col] = c.last_value[col] / c.init_cash[col] - 1
            
>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     entries=      pd.Series([X, O, O, O, O, O, O]),
...     exits=        pd.Series([O, O, O, X, O, O, O]),
...     short_entries=pd.Series([O, O, O, O, X, O, O]),
...     short_exits=  pd.Series([O, O, O, O, O, O, X]),
...     post_segment_func_nb=post_segment_func_nb,  # (5)!
...     in_outputs=dict(  # (6)!
...         returns=vbt.RepEval(  # (7)!
...             "np.full((target_shape[0], len(cs_group_lens)), np.nan)"
...         ),
...         total_return=vbt.RepEval(
...             "np.full(len(cs_group_lens), np.nan)"
...         )
...     )
... )
>>> pd.testing.assert_frame_equal(  # (8)!
...     pf.get_in_output("returns"),
...     pf.get_returns()
... )
>>> pd.testing.assert_series_equal(
...     pf.get_in_output("total_return"),
...     pf.get_total_return()
... )
```

1. [SignalSegmentContext.in_outputs](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SignalSegmentContext.in_outputs)
contains exactly the in-output tuple that we passed
2. If cash sharing is enabled, returns must be computed per group
3. If cash sharing is disabled, returns must be computed per column
4. Iterate over each column in the current group
5. Overriding this callback is enough to switch to the flexible mode
6. In-output tuple can be provided either as a named tuple or a dict. If the latter,
if will be automatically transformed into a named tuple by the method.
7. Use expression evaluation templates to build arrays only after the target shape is established.
Use `len(cs_group_lens)` to get the number of groups when cash sharing is enabled, and the number
of columns when cash sharing is disabled.
8. Check the pre-computed and reconstructed arrays for equality

The most impressive in the approach above is that any pre-computed metric that can be found
among the attributes of the [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) class 
is automatically picked by any built-in property and method that requires it, such as 
[Portfolio.stats](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.stats). The only limitation
is the inability to change the grouping after the array has been created; when this happens, 
the reconstructed version of the metric will be returned instead.

## Summary

Backtesting shouldn't be complex! One of the most crucial factors in successful trading is a proper 
entry and exit timing. By parametrizing the timing and direction of our trades, we can 
theoretically achieve the worst and best-possible returns in any market, thus the representation
of a trading strategy as a permutation of buy, sell, short-sell, and short-cover signals is
almost "[Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness)" in this regard. 
Reducing complex strategies into such a limited signal set has another advantage: we can enable and 
standardize backtesting of many trading strategies using the same piece of code. The method 
[Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
not only makes all of that possible, but it also encourages us to hack into the simulation to define
our custom trading logic iteratively, or to change the default simulation behavior, making it the 
best of both vectorized and event-driven worlds. But not everything is appropriate to be represented 
using signals: strategies that heavily rely upon variations in order information, such as rebalancing 
strategies, are best implemented using other means. Gladly, vectorbt offers solutions to such 
problems as well :ninja:


*[FO]: From-orders simulation method
*[FS]: From-signals simulation method
*[SL]: Stop loss
*[TSL]: Trailing stop loss
*[TTP]: Trailing take profit
*[TP]: Take profit

