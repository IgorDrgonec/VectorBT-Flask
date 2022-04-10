---
title: Portfolio
icon: material/chart-areaspline
---

# Portfolio

Portfolio refers to any combination of financial assets held by a trader. In the world of vectorbt,
"portfolio" is a multidimensional structure capable of simulating and tracking multiple independent but
also dependent portfolio instances. The main function of a portfolio is to apply a trading logic
on a set of inputs to simulate a realistic trading environment, also referred to as "simulation". 
The outputs of such a simulation are orders and other information that can be used by the user in 
assessing the portfolio's performance, also referred to as "reconstruction" or "post-analysis".
Both phases are isolated in nature, which enables various interesting use cases for quantitative 
analysis and data science.

The main class concerned with simulating and analyzing portfolios (i.e., with the actual backtesting) 
is [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio), which is a regular Python
class subclassing [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable)
and having a range of Numba-compiled functions at its disposal. It's built similarly to other
analyzable classes in the way that it has diverse class methods for instantiation from a range of 
inputs (such as [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
taking signals), it's a state-full class capable of wrapping and indexing any Pandas-like objects 
contained inside it, and it can compute metrics and display (sub-)plots for quick introspection 
of the stored data.

## Simulation

So, what's a simulation? It's just a sophisticated loop! :doughnut: 

A typical simulation in vectorbt takes some inputs (such as signals), gradually iterates over 
their rows (time steps in the real world) using a for-loop, and at each row, runs the trading 
logic by issuing and executing orders, and updating the current state of the trading environment 
such as the cash balance and position size. If we think about it, it's the exact same way we would
approach algorithmic trading in reality: at each minute/hour/day (= row), decide what to do 
(= trading logic), and place an order if you decided to change your position in the market.

Now, let's talk about execution. The core of the vectorbt's backtesting engine is fully Numba-compiled 
for best performance. The functionality of the engine is distributed across many functions in an 
entire sub-package - [portfolio.nb](/api/portfolio/nb/), ranging from core order execution commands 
to calculation of P&L in trade records. Remember that those functions aren't meant to be used 
directly (unless specifically desired) but are used by Python functions higher in the stack that 
know how to properly pre-process their input data and post-process the output data.

In the following parts, we'll discuss order execution and processing, and gradually implement
a collection of simple pipelines to better illustrate various simulation concepts.

### Primitive commands

Remember that vectorbt is an exceptionally raw backtester: it's primary commands are "buy" :green_circle: 
and "sell" :red_circle: This means that any strategy that can be translated into a set of those 
commands is also supported out of the box. This also means that more complex orders such as limit
and SL orders must be implemented manually. In contrast to other backtesting frameworks where processing 
is monolothic and functionality is written in an [object-oriented manner](https://en.wikipedia.org/wiki/Object-oriented_programming),
Numba forces vectorbt to implement most of the functionality as a spaghetti of functions :spaghetti:
(but don't worry - vectorbt was still designed with the best software design patterns in mind!)

!!! info
    Even though Numba supports OOP by compiling Python classes with `@jitclass`, they are treated
    as functions, must be statically typed, and have performance drawbacks that don't allow us
    to jump on the wagon just yet.

Functions related to order execution are primarily located in [portfolio.nb.core](/api/portfolio/nb/core/).
The functions implementing our primary two commands are 
[buy_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.buy_nb) and 
[sell_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.sell_nb). Among the requested size
and price of an order, the primary input of each of these functions is the current account 
state of type [AccountState](/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccountState), 
which contains the current cash balance, position size, and other information about the current 
environment. Whenever we buy or sell something, the function creates and returns a new state of the 
same type. Furthermore, it returns an order result of type [OrderResult](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderResult), 
which contains the filled size, price adjusted with slippage, transaction fee, order side, status
information on whether the order succeeded or failed, and valuable information about why it failed.

#### Buying

Let's say we have $100 available and want to buy 1 share at the price of $15:

```pycon
>>> import vectorbtpro as vbt
>>> from vectorbtpro.portfolio import nb as pf_nb, enums as pf_enums

>>> account_state = pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,  # (1)!
...     free_cash=100.0  # (2)!
... )
>>> new_account_state, order_result = pf_nb.buy_nb(
...     account_state=account_state,
...     size=1.0,
...     price=15.0
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=85.0,
    position=1.0,
    debt=0.0,
    free_cash=85.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=1.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

1. Debt is non-zero only when shorting
2. Free cash deviates from the regular cash balance only when shorting

The returned state indicates that we spent $15 and increased our position by 1 share. 
The order result contains details about the executed order: we bought 1 share for $15, 
with no transaction fees. Since order side and status are of named tuple type 
[OrderSide](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderSide) and 
[OrderStatus](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderStatus) respectively, 
we can query the meaning behind those numbers as follows:

```pycon
>>> pf_enums.OrderSide._fields[order_result.side]
'Buy'

>>> pf_enums.OrderStatus._fields[order_result.status]
'Filled'
```

!!! info
    If any value is `-1`, the information is unavailable.

Now, based on the new state, let's execute a transaction that uses up the remaining cash:

```pycon
>>> import numpy as np

>>> new_account_state2, order_result = pf_nb.buy_nb(
...     account_state=new_account_state,  # (1)!
...     size=np.inf,  # (2)!
...     price=15.0
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=0.0,
    position=6.666666666666667,
    debt=0.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=5.666666666666667,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

1. Use the previous account state as input
2. Infinity means using up the entire balance

Since vectorbt was originally tailored to cryptocurrency and fractional shares, the default 
behavior is buying as much as possible (here `5.67`), even if the amount is below of that requested. 
But what happens if we wanted to have the entire share instead? Let's specify the size 
granularity of 1, indicating that only integer amounts should be allowed:

```pycon
>>> new_account_state, order_result = pf_nb.buy_nb(
...     account_state,
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    free_cash=10.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=6.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

This has bought exactly 15 shares. Given the new account state, let's run the same transaction again:

```pycon
>>> new_account_state2, order_result = pf_nb.buy_nb(
...     new_account_state,
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    free_cash=10.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=1,
    status_info=5
)
```

The account state remains unchanged, while so many NaNs in the order result hint at a failed order.
Let's query the meaning behind the status and status information numbers using 
[OrderStatus](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderStatus) and 
[OrderStatusInfo](/api/portfolio/enums/#vectorbtpro.portfolio.enums.OrderStatusInfo) named tuple respectively:

```pycon
>>> pf_enums.OrderStatus._fields[order_result.status]
'Ignored'

>>> pf_enums.OrderStatusInfo._fields[order_result.status_info]
'SizeZero'

>>> pf_enums.status_info_desc[order_result.status_info]  # (1)!
'Size is zero'
```

1. There is a list with more elaborative descriptions of different status information

Here, the status "Size is zero" means that by considering our cash balance and after applying the 
size granularity, the (potentially) filled order size is zero, thus the order should be ignored.
Ignored orders have no effect on the trading environment and are simply, well, *ignored*. But sometimes,
when the user has specific requirements and vectorbt cannot execute them, the status will become "Rejected",
indicating that the request could not be fulfilled and an error can be thrown if wanted.

For example, let's try to buy more than possible:

```pycon
>>> new_account_state, order_result = pf_nb.buy_nb(
...     account_state=account_state, 
...     size=1000.0, 
...     price=15.0,
...     allow_partial=False
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=100.0,
    position=0.0,
    debt=0.0,
    free_cash=100.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=13
)

>>> pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> pf_enums.status_info_desc[order_result.status_info]
'Final size is less than requested'
```

There are many other parameters to control the execution. Let's use 50% of cash, and apply 1% 
in fees and slippage:

```pycon
>>> new_account_state, order_result = pf_nb.buy_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0,
...     fees=0.01,  # (1)!
...     slippage=0.01,  # (2)!
...     percent=0.5
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=50.0,
    position=3.2676534980230696,
    debt=0.0,
    free_cash=50.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=3.2676534980230696,
    price=15.15,
    fees=0.4950495049504937,
    side=0,
    status=0,
    status_info=-1
)
```

1. 0.01 = 1%. Paid always in cash. To specify fixed fees, use `fixed_fees` instead.
2. 0.01 = 1%. Applied on the price. By artificially increasing the price, you always put
yourself at a disadvantage, but this might be wanted to make backtesting more realistic.

The final fees and the price adjusted with the slippage are reflected in the order result.

Whenever we place an order, we can specify any price. Thus, it may sometimes happen that the 
provided price is (by mistake of the user) higher than the highest price of that bar or lower 
than the lowest price of that bar. Also, if the user wanted the price to be closing, and he specified
a slippage, this would also be quite unrealistic. To avoid such mistakes, the function performs an OHLC check.
For this, we need to specify the `price_area` of type [PriceArea](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceArea):
with the price boundaries, and what should be done if a boundary violation happens via `price_area_vio_mode`
of type [PriceAreaVioMode](/api/portfolio/enums/#vectorbtpro.portfolio.enums.PriceAreaVioMode):

```pycon
>>> price_area = pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> new_account_state, order_result = pf_nb.buy_nb(
...     account_state=account_state,
...     size=np.inf,
...     price=np.inf,
...     price_area=price_area,
...     price_area_vio_mode=pf_enums.PriceAreaVioMode.Error
... )
ValueError: Adjusted order price is above the highest price
```

#### Selling

The function for selling takes the same arguments but uses them in the opposite direction.
Let's remove 2 shares from a position of 10 shares:

```pycon
>>> account_state = pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     free_cash=0.0
... )
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state,
...     size=2.0,
...     price=15.0
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=30.0,
    position=8.0,
    debt=0.0,
    free_cash=30.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=2.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
```

The size in the order result remains positive but the side has changed from 0 to 1:

```pycon
>>> pf_enums.OrderSide._fields[order_result.side]
'Sell'
```

#### Shorting

Shorting is a regular sell operation with [sell_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.sell_nb),
but with one exception: it now involves the debt and free cash balance. Whenever we short, we are borrowing
shares and selling them to buyers willing to pay the market price. This operation increases the cash balance
and turns the position size negative. It also registers the received cash amount as a debt, and 
subtracts it from the free cash balance. Whenever we buy some shares back, the debt decreases 
proportionally to the value of the shares bought back, while the free cash might increase depending
upon whether the price was higher or lower than the average short-selling price. Whenever we cover 
the short position entirely, the debt becomes zero and the free cash balance returns to the same level 
as the regular cash balance.

!!! note
    You shouldn't treat debt as an absolute amount of cash you owe since you owe shares, not cash; 
    it's used only for calculating the average entry price of the short position, which is then used 
    to calculate the change in the free cash balance with each trade.

We are allowed to short sell **any amount**. By default, we need the same amount of funds in our margin 
account as the value of to-be-borrowed shares, that is, the available (free) cash acts as a collateral. 
Whenever this amount goes beyond the margin amount, the free cash balance becomes negative, without any 
consequences for us. But there are several instructions that we can use to instruct vectorbt to respect 
the initial margin requirement. 

The first one is by setting `lock_cash` to True, which will treat the free cash balance as a ceiling:

```pycon
>>> account_state = pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     free_cash=100.0
... )
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     lock_cash=True
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=200.0,
    position=-6.666666666666667,
    debt=100.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=6.666666666666667,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
```

The operation above doubled our regular cash balance, negated the position size, and used up
the entire free cash balance. 

Another option is to set the size to infinity and specify a percentage of the free cash to use:

```pycon
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     percent=1
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=200.0,
    position=-6.666666666666667,
    debt=100.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=6.666666666666667,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
```

!!! info
    Infinity is a special value in vectorbt and usually means "go as far as you can".

Let's try to run the same operation again, but now on the new account state:

```pycon
>>> new_account_state2, order_result = pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=np.inf, 
...     price=15.0, 
...     percent=1
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=200.0,
    position=-6.666666666666667,
    debt=100.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=6
)

>>> pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> pf_enums.status_info_desc[order_result.status_info]
'Not enough cash to short'
```

We see that vectorbt prevents the free cash balance to become negative.

Let's do a short sell without a ceiling by ordering twice the initial margin requirement:

```pycon
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     percent=2
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=300.0,
    position=-13.333333333333334,
    debt=200.0,
    free_cash=-100.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=13.333333333333334,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
```

If we had to calculate the current portfolio value, it would still default to the initial cash
since no transaction costs were involved and no additional trades were made:

```pycon
>>> new_account_state.cash + new_account_state.position * order_result.price
100.0
```

As we see, the positive cash balance and the negative position size keep the total value in balance.
Now, let's do another trade that tries to buy back 5 of the borrowed shares using 
[buy_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.buy_nb), given the price
has increased by 100% (my condolences):

```pycon
>>> new_account_state2, order_result = pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=30.0
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=150.0,
    position=-8.333333333333334,
    debt=125.0,
    free_cash=-100.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=5.0,
    price=30.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

Even though the regular cash balance has decreased (this is a buy operation after all), the free cash balance
hasn't changed, why so? Remember that only buying back with profit adds to the free cash balance.
But what definitely has changed is the debt, which decreased from $200 to $125 to keep the
average price per shorted share at $15:

```pycon
>>> new_account_state2.debt / abs(new_account_state2.position)
14.999999999999998
```

!!! important
    Having not exact numbers like this is an issue we have to live with when working
    with floating numbers. Thus, do not attempt to check whether the price is exactly $15.
    A native way to do it is by using special functions, such as 
    [is_close_nb](/api/utils/math_/#vectorbtpro.utils.math_.is_close_nb), that perform checks 
    within a tolerance.

    ```pycon
    >>> new_account_state2.debt / abs(new_account_state2.position) == 15
    False

    >>> from vectorbtpro.utils.math_ import is_close_nb
    >>> is_close_nb(new_account_state2.debt / abs(new_account_state2.position), 15)
    True
    ```

Let's say instead of jumping, the price has dipped to $10 per share (my congratulations!):

```pycon
>>> new_account_state2, order_result = pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=10.0
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=250.0,
    position=-8.333333333333334,
    debt=125.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=5.0,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

We see that the debt has decreased to the same level - $125, while the free cash balance is 0,
which means that the remaining short position now meets the initial margin requirements. Let's close
out the open short position using the same price:

```pycon
>>> new_account_state3, order_result = pf_nb.buy_nb(
...     account_state=new_account_state2, 
...     size=abs(new_account_state2.position), 
...     price=10.0
... )
>>> print(vbt.prettify(new_account_state3))
AccountState(
    cash=166.66666666666666,
    position=0.0,
    debt=0.0,
    free_cash=166.66666666666666
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=8.333333333333334,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

The free cash balance equals to the regular cash balance, and we are debt-free! Additionally,
those three operations have brought us $66.67 in profit.

#### Reversing

Positions in vectorbt can be reversed with a single order. Let's start with a position of 10 shares,
reverse it to the maximum extent in the short direction, and then reverse it to the maximum extent again 
but in the long direction:

```pycon
>>> account_state = pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     free_cash=0.0
... )
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     lock_cash=True
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=300.0,
    position=-10.0,
    debt=150.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)

>>> new_account_state2, order_result = pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> print(vbt.prettify(new_account_state2))
AccountState(
    cash=0.0,
    position=10.0,
    debt=0.0,
    free_cash=0.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

Both operations are symmetric in nature and cancel each other out by a repetitive call,
thus ultimately we've arrived at our initial state of the account.

#### Closing

To close out a position and to avoid its reversal, we can either specify the exact size, 
or the size of infinity and the current direction via the `direction` argument of type 
[Direction](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Direction).
For example, if we're in a long position and specified the long-only direction, the position 
won't be reversed:

```pycon
>>> account_state = pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     free_cash=0.0
... )
>>> new_account_state, order_result = pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     direction=pf_enums.Direction.LongOnly
... )
>>> print(vbt.prettify(new_account_state))
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    free_cash=150.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
```

!!! note
    Providing the opposite direction to the one our position is currently in doesn't mean that the 
    direction of the operation should be reversed, it simply means that vectorbt isn't allowed to
    enter a position of another direction. Using the [buy_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.buy_nb) 
    and [sell_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.sell_nb) command guarantees 
    to execute the order in the long and short direction respectively.

#### Pipeline/1

Even by using just those two essential commands, we can already build our own backtesting pipeline
of arbitrary complexity and flexibility. As said before, a simulation is just a loop that iterates 
over timestamps. Let's create a simplified pipeline that puts $1 into BiYFtcoin each time it discovers 
a [Golden Cross](https://www.investopedia.com/terms/g/goldencross.asp) entry signal, and sells $1 otherwise. 
What we want is one number: the final value of the portfolio.

```pycon
>>> from numba import njit

>>> @njit
... def pipeline_1_nb(close, entries, exits, init_cash=100):
...     account_state = pf_enums.AccountState(  # (1)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         free_cash=float(init_cash)
...     )
...     for i in range(close.shape[0]):
...         if entries[i]:
...             account_state, _ = pf_nb.buy_nb(  # (2)!
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...         if exits[i]:
...             account_state, _ = pf_nb.sell_nb(
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...     return account_state.cash + account_state.position * close[-1]  # (3)!

>>> data = vbt.YFData.fetch("BTC-USD", end="2022-01-01")
>>> sma_50 = vbt.talib("SMA").run(data.get("Close"), 50)
>>> sma_200 = vbt.talib("SMA").run(data.get("Close"), 200)
>>> entries = sma_50.real_crossed_above(sma_200)
>>> exits = sma_50.real_crossed_below(sma_200)

>>> pipeline_1_nb(
...     data.get("Close").values, 
...     entries.values, 
...     exits.values
... )
210.71073253390762
```

1. Initial account state
2. Execute the order and return a new account state
3. Calculate the final portfolio value

!!! hint
    Adding a suffix `_nb` to indicate a Numba-compiled function is not necessary 
    but still a good convention in vectorbt.

We can validate the pipeline using one of the preset simulation methods:

```pycon
>>> vbt.Portfolio.from_orders(
...     data.get("Close"), 
...     size=entries.astype(int) - exits.astype(int), 
...     size_type="value"
... ).final_value
210.71073253390762
```

### Order execution

Using the primitive commands is fun as long as we exactly know the direction of the order.
But very often, we have to deal with more complex requirements such as target percentages,
which change the order direction depending on the current value. In addition, the commands 
do not validate their arguments; for example, there won't be any error thrown
in a case when the user accidentally passes a negative order price. But also, we need a better
representation of an order - it's a bad practice of passing all the parameters such as slippage
as keyword arguments. 

All the checks and other pre-processing procedures are happening in the function
[execute_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.execute_order_nb).
The first input to this function is an order execution state of type 
[ExecState](/api/portfolio/enums/#vectorbtpro.portfolio.enums.ExecState), 
which contains the same information as an account state we saw above, but with additional 
information on the current valuation. The second input is a named tuple of type 
[Order](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Order) representing an order. 
The third argument is the price area, which we are also already familiar with.

#### Order

An order in vectorbt is represented by a [named tuple](https://realpython.com/python-namedtuple/).
Named tuples are alternatives to data classes in both the Python and Numba world; they are very 
efficient and lightweight data structures that can be easily constructed and processed.
Let's create an instance of an order:

```pycon
>>> order = pf_enums.Order()
>>> print(vbt.prettify(order))
Order(
    size=np.inf,
    price=np.inf,
    size_type=0,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=0.0,
    max_size=np.inf,
    size_granularity=np.nan,
    reject_prob=0.0,
    price_area_vio_mode=0,
    lock_cash=False,
    allow_partial=True,
    raise_reject=False,
    log=False
)
```

The tuple allows for attribute access through the dot notation:

```pycon
>>> order.direction
2

>>> order._fields  # (1)!
('size',
 'price',
 'size_type',
 'direction',
 'fees',
 'fixed_fees',
 'slippage',
 'min_size',
 'max_size',
 'size_granularity',
 'reject_prob',
 'price_area_vio_mode',
 'lock_cash',
 'allow_partial',
 'raise_reject',
 'log')
```

1. Get the field names

Other than this, it behaves just like any other tuple in Python:

```pycon
>>> order[3]
2

>>> tuple(order)  # (1)!
(inf,
 inf,
 0,
 2,
 0.0,
 0.0,
 0.0,
 0.0,
 inf,
 nan,
 0.0,
 0,
 False,
 True,
 False,
 False)
```

1. Convert to a regular tuple

One issue that we still have to address when working with Numba are default arguments:
although we can construct a new tuple solely with default arguments in Numba as we did above,
if we want to override some values, they must be located strictly on the left in that tuple's definition.
Otherwise, we must explicitly provide all the default arguments located before them:

```pycon
>>> @njit
... def create_order_nb():
...     return pf_enums.Order()  # (1)!

>>> create_order_nb()
Order(size=inf, price=inf, ..., log=False)

>>> @njit
... def create_order_nb(size, price):
...     return pf_enums.Order(size=size, price=price)  # (2)!

>>> create_order_nb(1, 15)
Order(size=1, price=15, ..., log=False)

>>> @njit
... def create_order_nb(size, price, direction):
...     return pf_enums.Order(size=size, price=price, direction=direction)  # (3)!

>>> create_order_nb(1, 15, 2)
Failed in nopython mode pipeline (step: nopython frontend)
```

1. Using the default values only
2. Overriding the default values of arguments on the left side
3. Overriding the default values of arguments on different positions

Another issue are data types. In the example above where we provided integer size and price,
Numba had no issues processing them. But as soon as we create such as order in a loop
and one of the arguments is a float instead of an integer provided previously, Numba will throw an error
because it cannot unify data types anymore. Thus, we should cast all arguments to their target data 
types before constructing an order.

Both issues are solved by using the function 
[order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.order_nb):

```pycon
>>> @njit
... def create_order_nb(size, price, direction):
...     return pf_nb.order_nb(size=size, price=price, direction=direction)

>>> create_order_nb(1, 15, 2)
Order(size=1.0, price=15.0, ..., log=False)
```

Notice how the size and price arguments were automatically cast to floats.

!!! hint
    Use [order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.order_nb)
    instead of [Order](/api/portfolio/enums/#vectorbtpro.portfolio.enums.Order) whenever possible.

#### Validation

Having constructed the order, [execute_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.execute_order_nb)
will check the arguments of that order for correct data types and values. For example, let's
try passing a negative price:

```pycon
>>> exec_state = pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> pf_nb.execute_order_nb(
...     exec_state,
...     pf_nb.order_nb(price=-15)
... )
ValueError: order.price must be finite and 0 or greater
```

#### Price resolution

After validating the inputs, [execute_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.execute_order_nb) 
uses them to decide which command to run: buy or sell. But first, it has to do some preprocessing.

Even though vectorbt isn't associated with any particular data schema and can run on tick data 
just as well as on bar data, it still gives us an option to provide the current candle (`price_area`) 
for validation and resolution reasons. In such a case, it will consider the passed order price
as a price point located within four price bounds: the opening, high, low, and closing price. 
Since order execution must happen strictly within those bounds, setting order price 
to `-np.inf` and `np.inf` will replace it by the opening and closing price respectively.
Hence, next time, when you see any default price being `np.inf`, just know that it means the 
close price :writing_hand:

```pycon
>>> price_area = pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> new_exec_state, order_result = pf_nb.execute_order_nb(  # (1)!
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=np.inf),
...     price_area=price_area
... )
>>> order_result.price
12.0

>>> new_exec_state, order_result = pf_nb.execute_order_nb(  # (2)!
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=-np.inf),
...     price_area=price_area
... )
>>> order_result.price
10.0

>>> new_exec_state, order_result = pf_nb.execute_order_nb(  # (3)!
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=-np.inf)
... )
>>> order_result.price
nan
```

1. Price gets replaced by the open price
2. Price gets replaced by the close price (default)
3. Price gets replaced by `np.nan` since the price area is not defined

#### Size type conversion

Our primitive commands accept only a size in the number of shares, thus we have to convert 
any size type defined in [SizeType](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SizeType) 
to `Amount`. Different size types require different information for conversion; for example, 
`TargetAmount` requires to know the current position size, while `Value` also requires to know 
the current valuation price.

Let's execute an order such that the new position has 3 shares:

```pycon
>>> new_exec_state, order_result = pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=pf_nb.order_nb(
...         size=3, 
...         size_type=pf_enums.SizeType.TargetAmount
...     ),
...     price_area=price_area
... )
>>> print(vbt.prettify(new_exec_state))
ExecState(
    cash=64.0,
    position=3.0,
    debt=0.0,
    free_cash=64.0,
    val_price=15.0,
    value=100.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=3.0,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

Since we're not in the market, vectorbt used [buy_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.buy_nb)
to buy 3 shares. If we were in the market with 10 shares, it would have used
[sell_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.sell_nb) to sell 7 shares.

#### Valuation

Speaking about the valuation price, it's the latest available price at the time of decision-making,
or the price used to calculate the portfolio value. In many simulation methods, valuation 
price defaults to the order price, but sometimes it makes more sense to use the open or previous 
close price for the conversion step. The separation of the valuation and order price enables us to 
introduce a time gap between order placement and its execution. This is important because, in reality, 
not always an order can be executed right away.

Let's order 100% of the portfolio value:

```pycon
>>> new_exec_state, order_result = pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=pf_nb.order_nb(
...         size=1.0, 
...         size_type=pf_enums.SizeType.TargetPercent
...     ),
...     price_area=price_area
... )
>>> print(vbt.prettify(new_exec_state))
ExecState(
    cash=20.0,
    position=6.666666666666667,
    debt=0.0,
    free_cash=20.0,
    val_price=15.0,
    value=100.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=6.666666666666667,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
```

Why haven't we spent the entire cash? Because to convert the target percentage into the target 
amount of shares, vectorbt used the provided order execution state with `val_price` of $15 and 
`value` of $100, which produced `100 / 15 = 6.67`. The closer the valuation price is to the order price, 
the closer the calculation result would be to the target requirement.

By default, if we want to place multiple orders within the same bar (for example, in pairs trading), 
vectorbt wouldn't adjust the portfolio value after each order. This is because it assumes that we made 
our trading decisions way before order execution and adjusting the value would affect those decisions. 
But also, an order has only a marginal immediate effect on the value, for example, because of a commission.
To force vectorbt to update the valuation price and value itself, we can enable `update_value`:

```pycon
>>> new_exec_state, order_result = pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=pf_nb.order_nb(
...         size=1.0, 
...         size_type=pf_enums.SizeType.TargetPercent,
...         fixed_fees=10,
...         slippage=0.01
...     ),
...     price_area=price_area,
...     update_value=True
... )
>>> print(vbt.prettify(new_exec_state))
ExecState(
    cash=9.199999999999989,
    position=6.666666666666667,
    debt=0.0,
    free_cash=9.199999999999989,
    val_price=12.120000000000001,
    value=90.0
)
>>> print(vbt.prettify(order_result))
OrderResult(
    size=6.666666666666667,
    price=12.120000000000001,
    fees=10.0,
    side=0,
    status=0,
    status_info=-1
)
```

Notice how the new valuation price has been set to the close price adjusted with the slippage
while the value has decreased by the fixed commission. Any new order placed after this one 
would use the updated value and thus probably produce a different outcome.

!!! note
    Use this feature only if you can control the order in which orders appear within a bar,
    and you have intra-bar data.

#### Pipeline/2

Let's create another simplified pipeline that orders given a target percentage array.
In particular, we'll keep 50% of the portfolio value in shares, and rebalance monthly.
We'll calculate the portfolio value based on the open price at the beginning of each bar, 
and order at the end of each bar (to keep things realistic). Also, we'll fill asset value 
and portfolio value arrays to later plot the allocation at each bar.

```pycon
>>> @njit
... def pipeline_2_nb(open, close, target_pct, init_cash=100):
...     asset_value_out = np.empty(close.shape, dtype=np.float_)  # (1)!
...     value_out = np.empty(close.shape, dtype=np.float_)
...     exec_state = pf_enums.ExecState(  # (2)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         free_cash=float(init_cash),
...         val_price=np.nan,
...         value=np.nan
...     )
...
...     for i in range(close.shape[0]):
...         if not np.isnan(target_pct[i]):  # (3)!
...             val_price = open[i]
...             value = exec_state.cash + val_price * exec_state.position  # (4)!
...
...             exec_state = pf_enums.ExecState(  # (5)!
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             order = pf_nb.order_nb(  # (6)!
...                 size=target_pct[i],
...                 price=close[i],
...                 size_type=pf_enums.SizeType.TargetPercent
...             )
...             exec_state, _ = pf_nb.execute_order_nb(  # (7)!
...                 exec_state=exec_state,
...                 order=order
...             )
...
...         asset_value_out[i] = exec_state.position * close[i]  # (8)!
...         value_out[i] = exec_state.cash + exec_state.position * close[i]
...         
...     return asset_value_out, value_out
```

1. Create two empty arrays with a floating data type. Remember that creating an array
with `np.empty` will produce an array with uninitialized (garbage) values that you should override.
2. Our initial order execution state
3. There is no need to run order execution when target percentage is `np.nan` (= do not rebalance)
4. Calculate the portfolio value at the beginning of the bar (= valuation)
5. Create a new existing order execution state after the valuation
6. Create a new order tuple using the close price as order price
7. Execute the order and return a new order execution state
8. Fill the arrays (you should fill each single element!)

Let's run the pipeline on our Bitcoin data:

```pycon
>>> symbol_wrapper = data.get_symbol_wrapper()  # (1)!
>>> target_pct = symbol_wrapper.fill()
>>> target_pct.vbt.set(0.5, every="MS", inplace=True)  # (2)!

>>> asset_value, value = pipeline_2_nb(
...     data.get("Open").values, 
...     data.get("Close").values, 
...     target_pct.values
... )
>>> asset_value = symbol_wrapper.wrap(asset_value)  # (3)!
>>> value = symbol_wrapper.wrap(value)
>>> allocations = (asset_value / value).rename(None)  # (4)!
>>> allocations.vbt.plot(trace_kwargs=dict(
...     mode="markers", 
...     marker=dict(
...         color=allocations, 
...         colorscale="Temps", 
...         size=3,
...         cmin=0.3,
...         cmid=0.5,
...         cmax=0.7
...     )
... ))
```

1. We cannot use `data.wrapper` because it contains OHLC as columns. What we need
is a wrapper that has symbols as columns, to fill the array with target percentages.
2. Fill the array with NaNs and set all data points at the beginning of each month to `0.5`.
3. Use the same wrapper to convert the NumPy array into Pandas Series
4. Divide the asset value by the portfolio value to derive the allocation

![](/assets/images/portfolio_2_allocation.svg)

!!! hint
    Each point represents a revaluation at the end of each bar.

We see that allocations are being regularly pulled back to the target level of 50%.

One of the biggest advantages of using vectorbt is that you can run your minimalistic trading 
environment in any Python function, even inside objective functions of machine learning models! 
There is no need to trigger the entire backtesting pipeline as a script or any other complex 
process like most backtesting frameworks force us to do :face_with_spiral_eyes:

### Order processing

Order execution takes an order instruction and translates it into a buy or sell operation.
The responsibility of the user is to do something with the returned order execution state and result;
mostly, we want to post-process and append each successful order to some list for later analysis - 
that's where order and log records come into play. Furthermore, we may want to raise an error if 
an order has been rejected and a certain flag in the requirements is present. All of this is ensured by 
[process_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.process_order_nb).

#### Order records

Order records is a [structured](https://numpy.org/doc/stable/user/basics.rec.html) NumPy array
of the data type [order_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.order_dt) containing 
information on each successful order. Each order in this array is assumed to be completed,
that is, you should view an order as a trade in the vectorbt's world. Since we're dealing with
Numba, we cannot and should not use lists and other inefficient data structures for storing
such complex information. Given that orders have fields with variable data types, the best 
data structure is a record array, which is a regular NumPy array with a complex data type and that 
behaves similarly to a Pandas DataFrame.

Since any NumPy array is a non-appendable structure, we should initialize an empty array of 
a sufficient size, and gradually fill it with new information. For this, we need a counter
- a simple integer - that points to an index of the next record to be written.

!!! info
    Actually, you can append to a NumPy array, but it will create a new array. Don't try this at home :smile:

Let's create an array with two order records and a counter:

```pycon
>>> order_records = np.empty(2, dtype=pf_enums.order_dt)
>>> order_cnt = 0
```

We shouldn't access this array just yet because it contains memory garbage, thus
it requires the user to manually set all the values in the array, and should be used with caution.

```pycon
array([(4585679916398730403, ..., 4583100142070297783),
       (4582795628349012822, ..., 4576866499094039639)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

Let's execute an order using [execute_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.execute_order_nb)
at the 678th bar, and fill the first record in the array:

```pycon
>>> exec_state = pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> new_exec_state, order_result = pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=15.0)
... )

>>> order_records["id"][order_cnt] = order_cnt  # (1)!
>>> order_records["col"][order_cnt] = 0
>>> order_records["idx"][order_cnt] = 678  # (2)!
>>> order_records["size"][order_cnt] = order_result.size
>>> order_records["price"][order_cnt] = order_result.price
>>> order_records["fees"][order_cnt] = order_result.fees
>>> order_records["side"][order_cnt] = order_result.side
>>> order_cnt += 1
```

1. Order ids start with 0 and follow the counter
2. Index of the current bar

!!! note
    When writing to an element of a record field, first select the field, and then the index.

At the next bar, we'll reverse the position and fill the second record:

```pycon
>>> new_exec_state2, order_result = pf_nb.execute_order_nb(
...     exec_state=new_exec_state,
...     order=pf_nb.order_nb(size=-np.inf, price=16.0)
... )

>>> order_records["id"][order_cnt] = order_cnt  # (1)!
>>> order_records["col"][order_cnt] = 0
>>> order_records["idx"][order_cnt] = 679
>>> order_records["size"][order_cnt] = order_result.size
>>> order_records["price"][order_cnt] = order_result.price
>>> order_records["fees"][order_cnt] = order_result.fees
>>> order_records["side"][order_cnt] = order_result.side
>>> order_cnt += 1
```

1. Don't forget to increment the order id

Here are the order records that we've populated:

```pycon
>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

But instead of setting each of these records manually, we can use 
[process_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.process_order_nb) to do it for us!
We just need to do one little adjustment: both the order records and the counter must be provided 
per column since vectorbt primarily works on multi-columnar data. This means that the order 
records array must become a two-dimensional array and the counter constant must become a one-dimensional array
(both with only one column in our case):

```pycon
>>> order_records = np.empty((2, 1), dtype=pf_enums.order_dt)
>>> order_counts = np.full(1, 0, dtype=np.int_)

>>> new_exec_state, order_result = pf_nb.process_order_nb(
...     0, 0, 678,  # (1)!
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=15.0),
...     order_records=order_records,
...     order_counts=order_counts
... )
>>> new_exec_state2, order_result = pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=pf_nb.order_nb(size=-np.inf, price=16.0),
...     order_records=order_records,
...     order_counts=order_counts
... )

>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
      
>>> order_counts
array([2])
```

1. Current group, column, and index

Such filled order records will become the backbone of the post-analysis phase.

#### Log records

Log records have the data type [log_dt](/api/portfolio/enums/#vectorbtpro.portfolio.enums.log_dt)
and are similar to order records, but with a few key differences: they are filled irrespective 
of whether the order has been filled, and they also contain information on the current execution 
state, the order request, and the new execution state. This way, we can completely and post-factum 
track down issues related to order processing.

```pycon
>>> log_records = np.empty((2, 1), dtype=pf_enums.log_dt)
>>> log_counts = np.full(1, 0, dtype=np.int_)

>>> new_exec_state, order_result = pf_nb.process_order_nb(
...     0, 0, 678,
...     exec_state=exec_state,
...     order=pf_nb.order_nb(size=np.inf, price=15.0, log=True),  # (1)!
...     log_records=log_records,
...     log_counts=log_counts
... )
>>> new_exec_state2, order_result = pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=pf_nb.order_nb(size=-np.inf, price=16.0, log=True),
...     log_records=log_records,
...     log_counts=log_counts
... )

>>> log_records
array([[(0, 0, 0, 678, ..., 0, 0, -1, -1)],
       [(1, 0, 0, 679, ..., 1, 0, -1, -1)]],
      dtype={'names':['id','group',...,'res_status_info','order_id'], ...})
```

1. Logging of each order must be explicitly enabled

!!! note
    Logging costs performance and memory. Use only when really needed.

#### Pipeline/3

Let's extend the [last pipeline](#pipeline2) to independently process an arbitrary number of columns, 
and gradually fill order records. This way, we can backtest multiple parameter combinations
by taking advantage of multidimensionality.

```pycon
>>> @njit
... def pipeline_3_nb(open, close, target_pct, init_cash=100):
...     order_records = np.empty(close.shape, dtype=pf_enums.order_dt)  # (1)!
...     order_counts = np.full(close.shape[1], 0, dtype=np.int_)
...
...     for col in range(close.shape[1]):  # (2)!
...         exec_state = pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             if not np.isnan(target_pct[i, col]):  # (3)!
...                 val_price = open[i, col]
...                 value = exec_state.cash + val_price * exec_state.position
...
...                 exec_state = pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     free_cash=exec_state.free_cash,
...                     val_price=val_price,
...                     value=value
...                 )
...                 order = pf_nb.order_nb(
...                     size=target_pct[i, col],
...                     price=close[i, col],
...                     size_type=pf_enums.SizeType.TargetPercent
...                 )
...                 exec_state, _ = pf_nb.process_order_nb(
...                     col, col, i,  # (4)!
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)  # (4)!
```

1. Since we don't know the number of orders in advance, let's prepare for the worst-case scenario:
one record at each bar. Remember that order records must be aligned column-wise.
2. Iterate over columns in `close` and run our logic on each one
3. Since every array passed to the pipeline now must be two-dimensional, don't forget to specify
the column when accessing an array element. Also, in indexing, first comes the row and then the column :point_up:
4. Use [repartition_nb](/api/generic/nb/base/#vectorbtpro.generic.nb.base.repartition_nb) to flatten
the final order records array (= concatenate records of all columns into a one-dimensional array)
5. Since all columns represent independent backtests, groups become columns

!!! info
    We are flattening (repartitioning) order records because most records are left unfilled,
    thus unnecessarily taking memory. By flattening, we're effectively compressing them
    without losing any information because each record already tracks the column it's supposed to be in.

Our pipeline now expects all arrays to be two-dimensional. Let's test three value combinations
of the parameter `every`, which controls the re-allocation periodicity. For this, we need
to expand all arrays to have the same number of columns as the parameter combinations.

```pycon
>>> import pandas as pd

>>> every = pd.Index(["MS", "Q", "Y"], name="every")

>>> open = data.get("Open").vbt.tile(3, keys=every)  # (1)!
>>> close = data.get("Close").vbt.tile(3, keys=every)
>>> close
every                                MS             Q             Y
Date                                                               
2014-09-17 00:00:00+00:00    457.334015    457.334015    457.334015
2014-09-18 00:00:00+00:00    424.440002    424.440002    424.440002
2014-09-19 00:00:00+00:00    394.795990    394.795990    394.795990
...                                 ...           ...           ...
2021-12-29 00:00:00+00:00  46444.710938  46444.710938  46444.710938
2021-12-30 00:00:00+00:00  47178.125000  47178.125000  47178.125000
2021-12-31 00:00:00+00:00  46306.445312  46306.445312  46306.445312

[2663 rows x 3 columns]

>>> target_pct = symbol_wrapper.fill().vbt.tile(3, keys=every)
>>> target_pct.vbt.set(0.5, every="MS", columns=["MS"], inplace=True)  # (2)!
>>> target_pct.vbt.set(0.5, every="Q", columns=["Q"], inplace=True)
>>> target_pct.vbt.set(0.5, every="Y", columns=["Y"], inplace=True)

>>> order_records = pipeline_3_nb(
...     open.values, 
...     close.values, 
...     target_pct.values
... )
>>> order_records
array([( 0, 0,   14, 1.29056570e-01,   383.61499023, 0., 0),  << first column
       ( 1, 0,   45, 1.00206092e-02,   325.74899292, 0., 0),
       ( 2, 0,   75, 7.10912824e-03,   379.24499512, 0., 1),
       ...
       (84, 0, 2571, 7.79003416e-04, 48116.94140625, 0., 0),
       (85, 0, 2602, 3.00678739e-03, 61004.40625   , 0., 1),
       (86, 0, 2632, 6.84410394e-04, 57229.828125  , 0., 0),
       ( 0, 1,   13, 1.32947604e-01,   386.94400024, 0., 0),  << second column
       ( 1, 1,  105, 1.16132613e-02,   320.19299316, 0., 0),
       ( 2, 1,  195, 1.83187063e-02,   244.22399902, 0., 0),
       ...
       (27, 1, 2478, 7.74416872e-03, 35040.8359375 , 0., 0),
       (28, 1, 2570, 2.08567037e-03, 43790.89453125, 0., 1),
       (29, 1, 2662, 1.72637091e-03, 46306.4453125 , 0., 1),
       ( 0, 2,  105, 1.60816173e-01,   320.19299316, 0., 0),  << third column
       ( 1, 2,  470, 2.34573523e-02,   430.56698608, 0., 1),
       ( 2, 2,  836, 3.81744650e-02,   963.74298096, 0., 1),
       ...
       ( 5, 2, 1931, 2.83026812e-02,  7193.59912109, 0., 1),
       ( 6, 2, 2297, 3.54188390e-02, 29001.72070312, 0., 1),
       ( 7, 2, 2662, 1.14541249e-02, 46306.4453125 , 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

1. Use [BaseAccessor.tile](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.tile)
to populate columns and append a new column level for our parameter combinations
2. Change the corresponding column only

This is exactly what [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)
requires as input: order records with a couple of other arrays can be used to reconstruct 
the simulation state, including the cash balance and the position size at each time step. 
We're slowly progressing towards post-analysis :slightly_smiling_face:

### Flexible indexing

The issue of bringing all arrays to the same shape as we did above is that it unnecessarily consumes memory:
even though the only array that has different data in each column is `target_pct`, we have
almost tripled memory consumption by having to expand other arrays like `close`. Imagine 
how expensive would it be having to align dozens of such array-like arguments :face_exhaling:

Flexible indexing allows us to overcome this alignment step and to access each element of an array
solely based on its shape. For example, there is no need to tile `close` three times if each 
row stays the same for each column - we can simply return the same row element irrespective of the column
being queried. The same goes for a one-dimensional array with elements per column - return the same column
element for each row. But how do we know whether a one-dimensional array has elements per column
or per row? This is solved by introducing another flag argument (`flex_2d`) that can switch the behavior.
The actual indexing is then done by the function 
[flex_select_auto_nb](/api/base/indexing/#vectorbtpro.base.indexing.flex_select_auto_nb), which
takes the array, the row, and the column being queried.

Let's demonstrate its use in different scenarios:

```pycon
>>> arr_0d = np.asarray(0)  # (1)!
>>> arr_1d = np.asarray([1, 2, 3])
>>> arr_2d = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

>>> vbt.flex_select_auto_nb(arr_0d, 1, 2)  # (2)!
0
>>> vbt.flex_select_auto_nb(arr_1d, 1, 2, flex_2d=False)  # (3)!
2
>>> vbt.flex_select_auto_nb(arr_1d, 1, 2, flex_2d=True)  # (4)!
3
>>> vbt.flex_select_auto_nb(arr_2d, 1, 2)  # (5)!
6
```

1. Constants must be instances of `np.ndarray` too
2. Considers each value in `arr_0d` to be defined per entire shape
3. Considers each value in `arr_1d` to be defined per row
4. Considers each value in `arr_1d` to be defined per column
5. Considers each value in `arr_2d` to be defined per element

Which yields the same result as if we had aligned the arrays prior to indexing (= memory expensive):

```pycon
>>> target_shape = (3, 3)

>>> np.broadcast_to(arr_0d, target_shape)[1, 2]
0
>>> np.broadcast_to(arr_1d[:, None], target_shape)[1, 2]
2
>>> np.broadcast_to(arr_1d, target_shape)[1, 2]
3
>>> np.broadcast_to(arr_2d, target_shape)[1, 2]
6
```

!!! hint
    `flex_2d` means _"Should the one-dimensional array broadcast against two dimensions or just one?"_ 
    Because broadcasting a one-dimensional array against two dimensions in NumPy will treat each 
    of its elements as per column rather than per row.

As a rule of thumb: when your entire simulation involves just one-dimensional data, set `flex_2d` to False.
Also set `flex_2d` to False if you iterate over multiple columns but all of your unaligned, one-dimensional 
data, such as `close`, is defined per row (timestamp). You can also leave the default `flex_2d`
and make all arguments two-dimensional, without tiling. This is the most proved way to avoid mistakes
and also consumes no additional memory:

```pycon
>>> vbt.flex_select_auto_nb(vbt.to_2d_array(arr_0d), 1, 2)
0
>>> vbt.flex_select_auto_nb(vbt.to_2d_array(arr_1d, expand_axis=1), 1, 2)
2
>>> vbt.flex_select_auto_nb(vbt.to_2d_array(arr_1d, expand_axis=0), 1, 2)
3
>>> vbt.flex_select_auto_nb(arr_2d, 1, 2)
6
```

#### Rotational indexing

But what happens if the index is out of bounds? Let's say we iterate over 6 columns but 
an array has data only for 3. In such a case, vectorbt can rotate the index and return
the first element in the array for the fourth column, the second element for the fifth column, and so on:

```pycon
>>> vbt.flex_select_auto_nb(arr_1d, 3, 4, flex_2d=False, rotate_rows=True)  # (1)!
1

>>> vbt.flex_select_auto_nb(arr_1d, 3, 4, flex_2d=True, rotate_cols=True)  # (2)!
2
```

1. Resolves to index 3 % 3 == 0 and element 1
2. Resolves to index 4 % 2 == 1 and element 2

If you think that this is crazy, and you would rather have an error shown: rotational indexing
is very useful when it comes to testing multiple assets and parameter combinations. Without it (default), 
we would need to tile the asset DataFrame by the number of parameter combinations, but with it, 
we could have just passed the data without tiling and thus wasting memory. But also, in many places, 
vectorbt ensures that all arrays can broadcast against other nicely anyway.

#### Pipeline/4

Let's adapt the previous pipeline for flexible indexing. Since usually we don't know which one of 
the passed arrays has the full shape, and sometimes there is no array with the full shape at all, 
we need to introduce another argument - `target_shape` - to provide the full shape for our loops to
iterate over. We'll also experiment with rotational indexing, which isn't supported by any of the 
preset simulation methods because the post-analysis phase requires the close price array to be of 
the full shape.

```pycon
>>> @njit
... def pipeline_4_nb(
...     target_shape, 
...     open, 
...     close, 
...     target_pct, 
...     init_cash=np.asarray(100),  # (1)!
...     flex_2d=False,
...     rotate_cols=False
... ):
...     order_records = np.empty(target_shape, dtype=pf_enums.order_dt)
...     order_counts = np.full(target_shape[1], 0, dtype=np.int_)
...
...     for col in range(target_shape[1]):
...         init_cash_elem = vbt.flex_select_auto_nb(
...             init_cash, col, rotate_cols=rotate_cols)  # (2)!
...
...         exec_state = pf_enums.ExecState(
...             cash=float(init_cash_elem),
...             position=0.0,
...             debt=0.0,
...             free_cash=float(init_cash_elem),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(target_shape[0]):
...             open_elem = vbt.flex_select_auto_nb(
...                 open, i, col, flex_2d, rotate_cols=rotate_cols)  # (3)!
...             close_elem = vbt.flex_select_auto_nb(
...                 close, i, col, flex_2d, rotate_cols=rotate_cols)
...             target_pct_elem = vbt.flex_select_auto_nb(
...                 target_pct, i, col, flex_2d, rotate_cols=rotate_cols)
...
...             if not np.isnan(target_pct_elem):
...                 value = exec_state.cash + open_elem * exec_state.position
...
...                 exec_state = pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     free_cash=exec_state.free_cash,
...                     val_price=open_elem,
...                     value=value
...                 )
...                 order = pf_nb.order_nb(
...                     size=target_pct_elem,
...                     price=close_elem,
...                     size_type=pf_enums.SizeType.TargetPercent
...                 )
...                 exec_state, _ = pf_nb.process_order_nb(
...                     col, col, i,
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

1. Initial cash must be provided per column, hence flexible indexing works here too
2. Select the current element of the initial cash array. Since it's always
a one-dimensional array with a column-wise layout, we will pass the current column as `i`
and disable `flex_2d` (we could have also passed the index as `col` and enabled `flex_2d`)
3. Since all three arrays are not guaranteed to have the full shape anymore,
we must switch to flexible indexing instead of doing `open[i, col]`

We can now use the original open and close arrays without tiling:

```pycon
>>> order_records = pipeline_4_nb(
...     target_pct.shape,  # (1)!
...     data.get("Open").values,  # (2)!
...     data.get("Close").values, 
...     target_pct.values
... )
>>> len(order_records)
125
```

1. Target percentage array holds the largest shape, thus use it for iteration
2. Keep other arrays one-dimensional

This also allows us to provide target percentages as a constant to re-allocate at each bar!

```pycon
>>> order_records = pipeline_4_nb(
...     symbol_wrapper.shape_2d,  # (1)!
...     data.get("Open").values,
...     data.get("Close").values, 
...     np.asarray(0.5)
... )
>>> len(order_records)
18641
```

1. Since we reduced the target percentage array to just a constant, our data 
now holds the largest shape, thus use it for iteration

This operation has generated the same number of orders as we have elements in the data:

```pycon
>>> np.product(symbol_wrapper.shape_2d)
18641
```

To demonstrate rotational indexing, let's pull multiple symbols and perform the simulation 
without having to tile or change them in any way:

```pycon
>>> mult_data = vbt.YFData.fetch(
...     ["BTC-USD", "ETH-USD"], 
...     end="2022-01-01",
...     missing_index="drop"
... )
```

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> mult_symbol_wrapper = mult_data.get_symbol_wrapper()
>>> target_pct = pd.concat([
...     mult_symbol_wrapper.fill().vbt.set(0.5, every=every[i])
...     for i in range(len(every))
... ], axis=1, keys=every)
>>> order_records = pipeline_4_nb(
...     target_pct.shape,  # (1)!
...     data.get("Open").values,  # (2)!
...     data.get("Close").values, 
...     target_pct.values,
...     rotate_cols=True
... )
>>> len(order_records)
142
```

1. That's another way of constructing the target percentage array
2. Target percentage array has again the largest shape
3. There is no need to tile any array thanks to rotational indexing

Without rotation, we would have got an _"IndexError: index out of bounds"_ error as the 
number of columns in the target shape is bigger than that in the price arrays.

### Grouping

Using groups, we can put multiple columns to the same backtesting basket :basket:

Generally, a group consists of a number of columns that are part of a single portfolio entity
and should be backtested as a single whole. Very often, we use groups to share capital among
multiple columns, but we can also use groups to bind columns on some logical level. During
a simulation, it's our responsibility to make use of grouping. For example, even though
[process_order_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.process_order_nb) requires
a group index, it uses it just for filling log records and nothing else. But after the simulation, 
vectorbt has many tools at its disposal to enable us in aggregating and analyzing various 
information per group, such as portfolio value.

Groups can be constructed and provided in two ways: as group lengths and as a group map. 
The former is easier to handle, marginally faster, and requires the columns to be split into monolithic 
groups, while the latter allows the columns of a group to be distributed arbitrarily, and is generally
a more flexible option. Group lengths is the format primarily used by simulation methods (since
asset columns, in contrast to parameter columns, are usually located next to each other),
while group maps are predominantly used by generic functions specialized in pre- and post-analysis. 
Both formats can be easily generated by a 
[Grouper](/api/base/grouping/base/#vectorbtpro.base.grouping.base.Grouper) instance.

#### Group lengths

Let's create a custom column index with 5 assets, and put them into 2 groups. Since group lengths
work on monolithic groups only, assets in each group must be next to each other:

```pycon
>>> columns = pd.Index(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"])
>>> mono_grouper = vbt.Grouper(columns, group_by=[0, 0, 0, 1, 1])
>>> mono_grouper.get_group_lens()  # (1)!
array([3, 2])
```

1. Using [Grouper.get_group_lens](/api/base/grouping/base/#vectorbtpro.base.grouping.base.Grouper.get_group_lens)

The first element in the returned array is the number of columns with the label `0`,
and the second element is the number of columns with the label `1`.

!!! hint
    [Grouper](/api/base/grouping/base/#vectorbtpro.base.grouping.base.Grouper) doesn't care
    if we pass a list of integers or a sequence of strings - it will convert everything into a 
    Pandas Index and treat it as group labels. They **don't** have to be alphanumerically sorted.

If we create discrete groups, the generation will fail:

```pycon
>>> dist_grouper = vbt.Grouper(columns, group_by=[0, 1, 0, 1, 1])
>>> dist_grouper.get_group_lens()
ValueError: group_by must form monolithic groups
```

Now, how do we define logic per group? Here's a template:

```pycon
>>> group_lens = mono_grouper.get_group_lens()

>>> group_end_idxs = np.cumsum(group_lens)  # (1)!
>>> group_start_idxs = group_end_idxs - group_lens  # (2)!

>>> for group in range(len(group_lens)):  # (3)!
...     from_col = group_start_idxs[group]
...     to_col = group_end_idxs[group]
...     # (4)!
...
...     for col in range(from_col, to_col):  # (5)!
...         pass  # (6)!
```

1. Get the end column index of each group (excluding)
2. Get the start column index of each group (including)
3. Iterate over all groups
4. Define here your logic per group
5. Iterate over all columns in the group
6. Define here your logic per column in the group

#### Group map

Group map is a tuple of two arrays:

1. One-dimensional array with column indices sorted by group
2. One-dimensional array with the length of each group in the first array

Thus, a group map makes distributed groups inherently monolithic, such that we can
work with any possible group distribution:

```pycon
>>> mono_grouper.get_group_map()
(array([0, 1, 2, 3, 4]), array([3, 2]))

>>> dist_grouper.get_group_map()
(array([0, 2, 1, 3, 4]), array([2, 3]))
```

In the second example, the first two (`2`) column indices in the first array belong to the first group,
while the remaining three (`3`) column indices belong to the second group.

Here's a template for working with a group map:

```pycon
>>> group_map = dist_grouper.get_group_map()

>>> group_idxs, group_lens = group_map
>>> group_start_idxs = np.cumsum(group_lens) - group_lens  # (1)!

>>> for group in range(len(group_lens)):
...     group_len = group_lens[group]
...     start_idx = group_start_idxs[group]
...     col_idxs = group_idxs[start_idx : start_idx + group_len]  # (2)!
...     # (3)!
... 
...     for k in range(len(col_idxs)):  # (4)!
...         col = col_idxs[k]
...         # (5)!
```

1. Get the start index of each group in the first array
2. Get the column indices of the group in the first array
3. Define here your logic per group
4. Iterate over all column indices in the group
5. Define here your logic per column in the group

#### Call sequence

When sharing capital between two or more assets, we sometimes want to process one column before the others.
This makes most sense, for example, in cases where we need to exit positions before opening new ones
to release funds for them. If we look at the templates for both grouping formats above, we can precisely
identify where the column processing order should be changed: in the for-loop that iterates over columns.
But how do we programmatically change this order? Here comes a call sequence into play.

Call sequence is an array of column indices in the order of their processing. For example,
if the third column should be processed first, the first column second, and the second column third,
the call sequence would be `[2, 0, 1]`. That is, we are always moving from left to right in 
the call sequence and pick the current column index. Such a design has one immense benefit:
we can use another array, such as with potential order values, to (arg-)sort the call sequence.

The sorting is done by the function [insert_argsort_nb](/api/utils/array_/#vectorbtpro.utils.array_.insert_argsort_nb),
which takes an array with values to sort by and an array of indices, and sorts the indices in-place
using [insertion sort](https://en.wikipedia.org/wiki/Insertion_sort) in the order the values appear 
in the first array. This sorting algorithm is best suited for smaller arrays and does not require any 
additional memory space - perfect for groups with assets!

Let's say we have three assets: one not in position, one in a short position, and one in a long position.
We want to close all positions in the order such that assets that should be sold are processed first.
Otherwise, we wouldn't have cash from exiting the long position to close out the short position.
For this, we will first use [approx_order_value_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.approx_order_value_nb)
to approximate the order value of each operation:

```pycon
>>> position = np.array([0.0, -10.0, 10.0])
>>> val_price = np.array([10.0, 5.0, 15.0])
>>> debt = np.array([0.0, 100.0, 0.0])
>>> order_value = np.empty(3, dtype=np.float_)

>>> for col in range(len(position)):
...     exec_state = pf_enums.ExecState(
...         cash=200.0,  # (1)!
...         position=position[col],  # (2)!
...         debt=debt[col],
...         free_cash=0.0,
...         val_price=val_price[col],
...         value=100.0
...     )
...     order_value[col] = pf_nb.approx_order_value_nb(
...         exec_state=exec_state,
...         size=0.,
...         size_type=pf_enums.SizeType.TargetAmount,
...         direction=pf_enums.Direction.Both
...     )
    
>>> order_value  # (3)!
array([0., 50., -150.])
```

1. Cash-related information is defined per group using a constant
2. Position-related information is defined per column using an array
3. Positive number means outbound cash flow, negative number means inbound cash flow

We see that the second column would require approx. $50 in cash and the third
column would bring approx. $150 in cash to close out the position. Let's create
a call sequence and sort it by the order value:

```pycon
>>> from vectorbtpro.utils.array_ import insert_argsort_nb

>>> call_seq = np.array([0, 1, 2])  # (1)!
>>> insert_argsort_nb(order_value, call_seq)
>>> call_seq
array([2, 0, 1])
```

1. We should always start with a simple range

!!! note
    Both the order value and the call sequence are sorted in-place!

We can then modify the for-loop to iterate over the call sequence instead:

```pycon
>>> for k in range(len(call_seq)):  # (1)!
...     c = call_seq[k]
...     col = from_col + c

>>> for k in range(len(call_seq)):  # (2)!
...     c = call_seq[k]
...     col = col_idxs[c]
```

1. When working with group lengths
2. When working with a group map

!!! hint
    A good practice is to keep a consistent naming of variables. Here, we're using 
    `k` to denote an index in the call sequence, `c` to denote a column index within a group, 
    and `col` to denote a global column index.

#### Pipeline/5

Let's upgrade our previous pipeline to rebalance groups of assets. To better illustrate how
important is sorting by order value when rebalancing multi-asset portfolios, we'll introduce another 
argument `auto_call_seq` to switch between sorting and not sorting. We will use group lengths
as the grouping format of choice because of its simplicity. Also note that now we have to keep
a lot of position-related information in arrays rather than constants since they exist
in relation to columns rather than groups. In addition, as we already know how to fill order records,
let's track the allocation at each bar instead.

```pycon
>>> @njit
... def pipeline_5_nb(
...     target_shape,  # (1)!
...     group_lens,  # (2)!
...     open,
...     close, 
...     target_pct,  # (3)!
...     init_cash=np.asarray(100),
...     auto_call_seq=True,
...     flex_2d=False,
...     rotate_cols=False
... ):
...     allocation = np.empty(target_shape, dtype=np.float_)  # (4)!
...
...     group_end_idxs = np.cumsum(group_lens)
...     group_start_idxs = group_end_idxs - group_lens
...
...     for group in range(len(group_lens)):  # (5)!
...         group_len = group_lens[group]
...         from_col = group_start_idxs[group]
...         to_col = group_end_idxs[group]
...
...         # (6)!
...         init_cash_elem = vbt.flex_select_auto_nb(
...             init_cash, group, rotate_cols=rotate_cols)
...     
...         last_position = np.full(group_len, 0.0, dtype=np.float_)  # (7)!
...         last_debt = np.full(group_len, 0.0, dtype=np.float_)
...         cash_now = float(init_cash_elem)
...         free_cash_now = float(init_cash_elem)
...
...         order_value = np.empty(group_len, dtype=np.float_)  # (8)!
...         call_seq = np.empty(group_len, dtype=np.int_)
... 
...         for i in range(target_shape[0]):  # (9)!
...             # (10)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_auto_nb(
...                     open, i, col, flex_2d, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * open_elem  # (11)!
...         
...             # (12)!
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_auto_nb(
...                     open, i, col, flex_2d, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_auto_nb(
...                     target_pct, i, col, flex_2d, rotate_cols=rotate_cols)
...                 exec_state = pf_enums.ExecState(
...                     cash=cash_now,
...                     position=last_position[c],
...                     debt=last_debt[c],
...                     free_cash=free_cash_now,
...                     val_price=open_elem,
...                     value=value_now,
...                 )
...                 order_value[c] = pf_nb.approx_order_value_nb(  # (13)!
...                     exec_state=exec_state,
...                     size=target_pct_elem,
...                     size_type=pf_enums.SizeType.TargetPercent,
...                     direction=pf_enums.Direction.Both
...                 )
...                 call_seq[c] = c  # (14)!
... 
...             if auto_call_seq:
...                 pf_nb.insert_argsort_nb(order_value, call_seq)  # (15)!
... 
...             for k in range(len(call_seq)):  # (16)!
...                 c = call_seq[k]  # (17)!
...                 col = from_col + c  # (18)!
...
...                 open_elem = vbt.flex_select_auto_nb(
...                     open, i, col, flex_2d, rotate_cols=rotate_cols)
...                 close_elem = vbt.flex_select_auto_nb(
...                     close, i, col, flex_2d, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_auto_nb(
...                     target_pct, i, col, flex_2d, rotate_cols=rotate_cols)
...
...                 if not np.isnan(target_pct_elem):  # (19)!
...                     order = pf_nb.order_nb(
...                         size=target_pct_elem,
...                         price=close_elem,
...                         size_type=pf_enums.SizeType.TargetPercent,
...                         direction=pf_enums.Direction.Both
...                     )
...                     exec_state = pf_enums.ExecState(
...                         cash=cash_now,
...                         position=last_position[c],
...                         debt=last_debt[c],
...                         free_cash=free_cash_now,
...                         val_price=open_elem,
...                         value=value_now,
...                     )
...                     new_exec_state, order_result = pf_nb.process_order_nb(
...                         group=group,
...                         col=col,
...                         i=i,
...                         exec_state=exec_state,
...                         order=order
...                     )
...                     cash_now = new_exec_state.cash
...                     free_cash_now = new_exec_state.free_cash
...                     value_now = new_exec_state.value
...                     last_position[c] = new_exec_state.position
...                     last_debt[c] = new_exec_state.debt
...
...             # (20)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_auto_nb(
...                     close, i, col, flex_2d, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * close_elem
...
...             # (21)!
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_auto_nb(
...                     close, i, col, flex_2d, rotate_cols=rotate_cols)
...                 allocation[i, col] = last_position[c] * close_elem / value_now
... 
...     return allocation
```

1. Second number in the target shape tracks assets (x parameter combinations) - it doesn't track groups!
2. The group lengths array must have the same number of elements as we have groups, while
the sum of this array must yield the number of columns in `target_shape`
3. Target allocations must be provided per asset, thus the array should broadcast against `target_shape`
4. Allocations must be filled per asset, thus the same shape as `target_shape`
5. Iterate over groups
6. Here comes the creation of various arrays that should exist only per group,
such as the cash balance, position size per asset, and other state information. Remember that different 
groups represent independent, isolated tests, and shouldn't be connected by any means!
7. We can't create a single instance of [ExecState](/api/portfolio/enums/#vectorbtpro.portfolio.enums.ExecState)
like we did before because an order execution state contains information per asset, thus
we need to keep track of its fields using separate variables (constants for data per group, 
arrays for data per asset)
8. We could have also created those two arrays at each bar, but frequent array creation
slows down the simulation. A better practice is to create an array only once and re-fill it 
as many times as we want.
9. Iterate over time steps (bars) as we did in previous pipelines
10. Calculate the value of each group (= portfolio value) by iterating over the assets of the group
and adding their position value to the current cash balance
11. Last position array exists only for this group, thus we're using `c`, not `col`!
12. Prepare the order value and call sequence arrays
13. Approximate the value of a potential order. We're at the beginning of the bar, thus we use the open price.
14. Write the current column index within this group. This will connect order values to column indices.
15. Sort both arrays in-place such that column indices associated with a lower order value appear first
in the call sequence
16. Next we want to execute orders, hence iterate over the newly sorted call sequence
17. Get the associated column index within the group
18. Get the associated global column index
19. Perform the same logic as in the previous pipeline
20. Calculate the group value once again, but now using the updated position and the close price.
Note that we are running this outside of the `np.nan` check since we want to track the allocation
at each single bar.
21. Calculate the current allocation of each asset, and write it to the output array

Wow, this went complex really fast! :dizzy_face:

But it's not that complex as it may appear. We took a bunch of columns and split them into 
groups. Then, for each group, we defined a mini-pipeline that applies our logic on the columns 
within this group only, acting as a single portfolio unit. At the beginning of each bar, we calculate 
the portfolio value, and build a call sequence that re-arranges columns by their order value. We then 
iterate over this sequence and execute an order in each column. Finally, at the end of each bar, 
we again re-calculate the portfolio value and write the real allocation of each asset to the output array.
The best in this pipeline is that it closely mimics how preset simulation methods work in vectorbt,
and it's one of the most flexible pieces of code you can actually write!

Let's allocate 70% to BTC and 30% to ETH, and rebalance on a monthly basis:

```pycon
>>> target_pct = mult_symbol_wrapper.fill()
>>> target_pct.vbt.set([[0.7, 0.3]], every="MS", inplace=True)
>>> grouper = vbt.Grouper(mult_symbol_wrapper.columns, group_by=True)
>>> group_lens = grouper.get_group_lens()

>>> allocation = pipeline_5_nb(
...     target_pct.shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     target_pct.values
... )
>>> allocation = mult_symbol_wrapper.wrap(allocation)
>>> allocation.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... )
```

![](/assets/images/pipeline_5_auto_call_seq.svg)

!!! info
    As you might have noticed, some allocations do not quite sum to 100%. This is because we used the 
    open price for group valuation and decision-making, while the actual orders were executed using
    the close price. By the way, it's a bad sign when everything aligns perfectly - this could mean 
    that your simulation is too ideal for the real world.

And here's the same procedure but without sorting the call sequence array:

```pycon
>>> allocation = pipeline_5_nb(
...     target_pct.shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     target_pct.values,
...     auto_call_seq=False
... )
>>> allocation = mult_symbol_wrapper.wrap(allocation)
>>> allocation.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... )
```

![](/assets/images/pipeline_5_wo_auto_call_seq.svg)

As we see, some rebalancing steps couldn't be completed at all because long operations
were executed before short operations, leaving them without the required funds.

The biggest advantage of this pipeline is in its flexibility: we can turn off grouping via `group_by=False` 
to run the entire logic per column (each group will contain only one column). We can also test 
multiple weight combinations via multiple groups, without having to tile the pricing data 
thanks to rotational indexing. This, for example, can't be done even with 
[Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) :wink:

```pycon
>>> groups = pd.Index([0, 0, 1, 1], name="group")
>>> target_alloc = pd.Index([0.7, 0.3, 0.5, 0.5], name="target_alloc")

>>> final_columns = vbt.stack_indexes((  # (1)!
...     groups,
...     target_alloc, 
...     vbt.tile_index(mult_symbol_wrapper.columns, 2)
... ))
>>> final_wrapper = mult_symbol_wrapper.replace(  # (2)!
...     columns=final_columns,
...     group_by="group"
... )
>>> target_pct = final_wrapper.fill(group_by=False)  # (3)!
>>> target_pct.vbt.set(target_alloc.values[None], every="MS", inplace=True)
>>> group_lens = final_wrapper.grouper.get_group_lens()

>>> allocation = pipeline_5_nb(
...     target_pct.shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     target_pct.values,
...     rotate_cols=True
... )
>>> allocation = target_pct.vbt.wrapper.wrap(allocation)
>>> allocation
group                                       0                   1          
target_alloc                    0.7       0.3       0.5       0.5
symbol                      BTC-USD   ETH-USD   BTC-USD   ETH-USD
Date                                                             
2017-11-09 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-10 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-11 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-12 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-13 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
...                             ...       ...       ...       ...
2021-12-27 00:00:00+00:00  0.703817  0.296183  0.504464  0.495536
2021-12-28 00:00:00+00:00  0.703452  0.296548  0.504026  0.495974
2021-12-29 00:00:00+00:00  0.708035  0.291965  0.509543  0.490457
2021-12-30 00:00:00+00:00  0.706467  0.293533  0.507650  0.492350
2021-12-31 00:00:00+00:00  0.704346  0.295654  0.505099  0.494901

[1514 rows x 4 columns]
```

1. Build a new column hierarchy with three levels: groups, weights, and assets. 
Each level must have the same length.
2. Create a new (grouped) wrapper with the new column hierarchy
3. Fill the target percentage array using the new wrapper

### Contexts

Sometimes, there is a need to create a simulation method that takes a user-defined function and calls it 
to make some trading decision. Such a UDF would require access to the simulation's state (such as the 
current position size and direction) and other information, which could quickly involve dozens 
of variables. Remember that we cannot do full-scale OOP in Numba, thus we have to pass data using
primitive containers such as tuples. But usage of variable positional arguments or a regular tuple
would be quite cumbersome for the user because accessing each field can only be done using an 
integer index or tuple unpacking. To ease this burden, we usually pass such information in form of 
a named tuple, often referred to as a (simulation) "context".

#### Pipeline/6

Let's create a very basic pipeline that iterates over rows and columns, and, at each element, 
calls a UDF to get an order and execute it!

First, we need to answer the following question: "What information would a UDF need?" 
Mostly, we just include everything we have:

```pycon
>>> from collections import namedtuple

>>> SimContext = namedtuple("SimContext", [
...     "open",  # (1)!
...     "high",
...     "low",
...     "close",
...     "init_cash",
...     "col",  # (2)!
...     "i",
...     "price_area",  # (3)!
...     "exec_state"
... ])
```

1. Arguments passed to the simulator
2. Loop variables
3. State information, either unpacked (marginally faster) or in form of named tuples (more convenient)

And here's our pipeline that takes and calls an order function:

```pycon
>>> @njit
... def pipeline_6_nb(
...     open, high, low, close, 
...     order_func_nb, order_args=(), 
...     init_cash=100
... ):
...     order_records = np.empty(close.shape, dtype=pf_enums.order_dt)
...     order_counts = np.full(close.shape[1], 0, dtype=np.int_)
...
...     for col in range(close.shape[1]):
...         exec_state = pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             val_price = open[i, col]
...             value = exec_state.cash + val_price * exec_state.position
...
...             price_area = pf_enums.PriceArea(
...                 open[i, col],
...                 high[i, col],
...                 low[i, col],
...                 close[i, col]
...             )
...             exec_state = pf_enums.ExecState(
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             sim_ctx = SimContext(  # (1)!
...                 open=open,
...                 high=high,
...                 low=low,
...                 close=close,
...                 init_cash=init_cash,
...                 col=col,
...                 i=i,
...                 price_area=price_area,
...                 exec_state=exec_state
...             )
...             order = order_func_nb(sim_ctx, *order_args)  # (2)!
...             exec_state, _ = pf_nb.process_order_nb(
...                 col, col, i,
...                 exec_state=exec_state,
...                 order=order,
...                 price_area=price_area,
...                 order_records=order_records,
...                 order_counts=order_counts
...             )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

1. Initialize the simulation context (= creates a named tuple)
2. Call the UDF by first passing the context and then any user-defined arguments

Let's write our own order function that generates orders based on signals:

```pycon
>>> @njit  # (1)!
... def signal_order_func_nb(c, entries, exits):
...     if entries[c.i, c.col] and c.exec_state.position == 0:  # (2)!
...         return pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return pf_nb.close_position_nb()
...     return pf_nb.order_nothing_nb()

>>> pipeline_6_nb(
...     vbt.to_2d_array(data.get("Open")),  # (3)!
...     vbt.to_2d_array(data.get("High")), 
...     vbt.to_2d_array(data.get("Low")), 
...     vbt.to_2d_array(data.get("Close")), 
...     signal_order_func_nb,
...     order_args=(
...         vbt.to_2d_array(entries),  # (4)!
...         vbt.to_2d_array(exits)
...     )
... )
array([( 0, 0,  300, 0.34786966,   287.46398926, 0., 0),
       ( 1, 0,  362, 0.34786966,   230.64399719, 0., 1),
       ( 2, 0,  406, 0.26339233,   304.61801147, 0., 0),
       ( 3, 0, 1290, 0.26339233,  6890.52001953, 0., 1),
       ( 4, 0, 1680, 0.33210511,  5464.86669922, 0., 0),
       ( 5, 0, 1865, 0.33210511,  9244.97265625, 0., 1),
       ( 6, 0, 1981, 0.31871477,  9633.38671875, 0., 0),
       ( 7, 0, 2016, 0.31871477,  6681.06298828, 0., 1),
       ( 8, 0, 2073, 0.2344648 ,  9081.76171875, 0., 0),
       ( 9, 0, 2467, 0.2344648 , 35615.87109375, 0., 1),
       (10, 0, 2555, 0.17333543, 48176.34765625, 0., 0)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

1. Don't forget to decorate your order function with `@njit` as well!
2. We can access any information similarly to accessing attributes of any Python object
3. Since we also iterate over columns, don't forget to expand any one-dimensional data
to two dimensions, easily using [to_2d_array](/api/base/reshaping/#vectorbtpro.base.reshaping.to_2d_array)
4. These two arrays were generated for the [first pipeline](#pipeline1)

We just created our own shallow [Portfolio.from_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_order_func)
functionality, neat! :boom:

### Performance

In terms of performance, Numba code is often a roller coaster :roller_coaster:

Numba is a just-in-time (JIT) compiler that analyzes and optimizes code, and finally uses the 
[LLVM compiler library](https://github.com/numba/llvmlite) to generate a machine code version of a 
Python function to be compiled. But sometimes, even if the function looks efficient on paper, Numba may
generate a suboptimal machine code because of some variables or their types not interacting optimally.
In such a case, the code may still run very fast compared to a similar implementation with Python
or even to another JIT compiler, but there is a lot of space for improvement that may be hard to discover,
even for experienced users. There are even cases where switching the lines in which variables are defined
suddenly and unexpectedly has a negative/positive effect on performance.

Apart from [official tips](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html), 
there are some of the best practices you should always keep in mind when designing and optimizing 
Numba-compiled functions:

1. Numba is perfectly happy with loops, and often even more happy than with vectorized operations.
That's why 90% of vectorbt's functionality is enabled by loops.
2. Numba hates repeated creation of new arrays and allocating (even small) chunks of memory in loops. 
A much better idea is to create a bunch of bigger arrays prior to the iteration, and use them as a buffer
for storing temporary information. Be aware that operations with NumPy that yield a new array, 
such as `np.cumsum`, create a new array!
3. Reading and writing one array element at a time is more efficient than in chunks
4. Basic math operations such as `-`, `+`, `*`, and `/` should be preferred to NumPy operations
5. When using a function as an argument to another function, arguments to this function should
be accepted in a packed format (`args`) instead of an unpacked format (`*args`). This rule is
often violated by vectorbt itself, but such cases are usually benchmarked to ensure that 
performance stays the same.
6. Packing named tuples inside other named tuples (as we did above) is not encouraged,
but sometimes there is no negative effect at all
7. NumPy arrays are almost always a better deal than lists and dictionaries!
8. Even if `fastmath=True` option has a positive impact on performance, be aware
that it's associated with various [compromises](https://llvm.org/docs/LangRef.html#fast-math-flags) 
when doing numeric operations
9. Do not iterate over elements of an array, iterate over a range with the same length instead
and use the loop variable to select the respective element
10. When overwriting a variable, make sure that it has the same type

!!! hint
    As a rule of thumb: the simpler is the code, the easier it becomes for Numba to analyze and optimize it.

#### Benchmarking

To benchmark a simulator, we can use the [timeit](https://docs.python.org/3/library/timeit.html) module.
If possible, create some sample data of a sufficient size, and prepare for the worst-case scenario
where orders are issued and executed at each single time step to benchmark the full load.
Also, make sure to run tests all the way during the simulator's development to track the evolution
of its execution time and stability.

!!! note
    Generation of sample data and preparation of other inputs must be done prior to benchmarking.

Let's generate 1-minute random OHLC data for one year using 
[RandomOHLCData](/api/data/custom/#vectorbtpro.data.custom.RandomOHLCData):

```pycon
>>> test_data = vbt.RandomOHLCData.fetch(
...     start="2020-01-01", 
...     end="2021-01-01",
...     freq="1s",  # (1)!
...     std=0.0001,  # (2)!
...     symmetric=True,  # (3)!
...     seed=41,
...     ohlc_freq="1min"  # (4)!
... )
>>> test_data.resample("1d").plot()  # (5)!
```

1. Set tick frequency to 1 second
2. Set tick volatility to 0.01%
3. Use symmetric returns (50% negative return == 100% positive return)
4. Resample to 1-minute OHLC
5. Plot to ensure that the generated data is realistic. Here we're resampling to daily
frequency for faster plotting.

![](/assets/images/simulation_random_ohlc_data.svg)

Then, we need to prepare all the data, which includes filling signals such that there is at least one 
order at each bar (our worst-case scenario for performance and memory):

```pycon
>>> test_open = test_data.get("Open").values[:, None]  # (2)!
>>> test_high = test_data.get("High").values[:, None]
>>> test_low = test_data.get("Low").values[:, None]
>>> test_close = test_data.get("Close").values[:, None]
>>> test_entries = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_exits = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_entries[0::2] = True  # (3)!
>>> test_exits[1::2] = True  # (4)!
>>> del test_data  # (5)!

>>> print(test_entries.shape)
(527041, 1)
```

1. Generate random OHLC data with symmetric returns
2. Get a column, extract the NumPy array, and expand the array to two dimensions (omit the
last step if your data is already two-dimensional!)
3. Place an entry at each second bar starting from the first bar
4. Place an exit at each second bar starting from the second bar
5. Delete the data object to release memory

Each of the arrays is 527,041 data points long. 

So, how is our simulator performing on this data?

```pycon
>>> %%timeit  # (1)!
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
79.4 ms ± 290 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

1. This magic command works only in a Jupyter environment and only if you place this command 
at the beginning of the cell. If you're running the code as a script, use the `timeit` module.

80 milliseconds to generate half a million orders on Apple M1, not bad! :fire:

To better illustrate how only a minor change can impact performance, we will create a new
order function that also creates a zero-sized empty array:

```pycon
>>> @njit
... def subopt_signal_order_func_nb(c, entries, exits):
...     _ = np.empty(0)  # (1)!
...
...     if entries[c.i, c.col] and c.exec_state.position == 0:
...         return pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return pf_nb.close_position_nb()
...     return pf_nb.order_nothing_nb()

>>> %%timeit
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     subopt_signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
130 ms ± 675 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

1. Here

As we see, creating an empty array at each bar has slowed down the execution by more than 50%.
And this is a very important lesson to learn: create arrays outside of loops and only once!

#### Auto-parallelization

Because of path dependencies (= the current state depends on the previous one), we cannot parallelize 
the loop that iterates over rows (= time). But here's the deal: since vectorbt allows us
to define a multi-columnar backtesting logic, we can parallelize the loop that iterates over 
columns or groups of columns, given that those columns or groups of columns are independent 
of each other - all using Numba alone. By the way, this is one of the primary reasons why vectorbt 
loves two-dimensional data layouts so much.

Automatic parallelization with Numba cannot be simpler: just replace `range` that you want to 
parallelize with `numba.prange`, and instruct Numba to parallelize the function by passing 
`parallel=True` to the `@njit` decorator. This will (try to) execute the code in the loop simultaneously 
by multiple parallel threads. You can read more about automatic parallelization with Numba 
[here](https://numba.pydata.org/numba-doc/latest/user/parallel.html) and about the
available threading layers [here](https://numba.pydata.org/numba-doc/latest/user/threading-layer.html). 
On MacBook Air (M1, 2020), turning on parallelization reduces the processing time by 2-3 times on average.
Usually, a simple arithmetic-heavy code without creating any arrays can be better parallelized 
than a complex vectorization-heavy code.

!!! important
    You can modify the same array from multiple threads, as done by countless functions in vectorbt.
    Just make sure that multiple threads (columns, in our case) aren't modifying the same elements
    and data in general!

Here's a small example of a function that computes the expanding maximum on two-dimensional data,
without and with automatic parallelization:

```pycon
>>> rom numba import prange

>>> arr = np.random.uniform(size=(1000000, 10))

>>> @njit
... def expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=np.float_)
...     for col in range(arr.shape[1]):
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit expanding_max_nb(arr)
40.7 ms ± 558 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> @njit(parallel=True)  # (1)!
... def parallel_expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=np.float_)
...     for col in prange(arr.shape[1]):  # (2)!
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit parallel_expanding_max_nb(arr)
26.6 ms ± 437 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

1. Here's the first change
2. Here's the second change

It's your turn: enable automatic parallelization of columns in the [sixth pipeline](#pipeline6) 
and benchmark it! Just don't forget to reduce the number of rows and increase the number of columns.

#### Caching

Even if we had optimized the simulation pipeline for the best-possible performance, the actual compilation
step would take a huge chunk of that time savings away. However, the good news is that
Numba doesn't have to re-compile the function the second time it's executed, given that we passed 
the same argument **types** (not data!). This means that we need to wait only once if we want to test the 
same function on many parameter combinations, at the same Python runtime. Sadly, if only one argument 
differs in type, or we've restarted the Python runtime, Numba has to compile again. 

But luckily, Numba gives us a mechanism to avoid re-compilation even if we've restarted the runtime, 
called [caching](https://numba.pydata.org/numba-doc/latest/developer/caching.html).
To enable caching, just pass `cache=True` to the `@njit` decorator.

!!! important
    Avoid turning on caching for functions that take complex, user-defined data, such as
    (named) tuples and other functions. This may lead to some hidden bugs and kernel crashes if the 
    data changes during the next runtime. Also make sure that your function doesn't use global variables.
    For example, the [fifth pipeline](#pipeline5) is perfectly cacheable, while the [sixth pipeline](#pipeline6)
    is not cacheable, or maybe could be if `order_func_nb` was cacheable as well.

#### AOT compilation

Using [ahead-of-time compilation](https://numba.pydata.org/numba-doc/dev/user/pycc.html), we can compile
a function only once and get no compilation overhead at runtime. Although this feature of Numba
isn't widely used in vectorbt because it would restrict us from passing input data flexibly, we
can make use of it in cases where we know the argument types in advance. Let's pre-compile our
[fifth pipeline](#pipeline5)!

For this, we have to specify the signature of a function explicitly. You can read more about it in the 
[types](https://numba.pydata.org/numba-doc/dev/reference/types.html#numba-types) reference.

```pycon
>>> from numba.pycc import CC
>>> cc = CC('pipeline_5')  # (1)!

>>> sig = "f8[:, :](" \ # (2)!
...       "UniTuple(i8, 2), " \ # (3)!
...       "i8[:], " \ # (4)!
...       "f8[:, :], " \ # (5)!
...       "f8[:, :], " \
...       "f8[:, :], " \
...       "f8[:], " \ # (6)!
...       "b1, " \ # (7)!
...       "b1, " \
...       "b1" \
...       ")"

>>> cc.export('pipeline_5_nb', sig)(pipeline_5_nb)   # (8)!
>>> cc.compile() # (9)!
```

1. Initialize a new module
2. Function should return a two-dimensional array of 64-bit floating-point data type (allocations)
3. Tuple with two 64-bit integers (`target_shape`)
4. One-dimensional array of 64-bit integer data type (`group_lens`)
5. Two-dimensional array of 64-bit floating-point data type (`open`, `close`, and `target_pct`)
6. One-dimensional array of 64-bit floating-point data type (`init_cash`)
7. Boolean value (`auto_call_seq`, `flex_2d`, and `rotate_cols`)
8. Register the function with the provided signature. You can bind multiple signatures
to the same function.
9. Compile the module

This has generated an extension module named `pipeline_5`. On macOS, the actual filename is 
`pipeline_5.cpython-37m-darwin.so`. We can then import the module like a regular Python module
and run the function `pipeline_5_nb` of that module:

```pycon
>>> import pipeline_5

>>> target_pct = mult_symbol_wrapper.fill()
>>> target_pct.vbt.set([[0.7, 0.3]], every="MS", inplace=True)
>>> grouper = vbt.Grouper(mult_symbol_wrapper.columns, group_by=True)
>>> group_lens = grouper.get_group_lens()
>>> allocation = pipeline_5.pipeline_5_nb(
...     target_pct.shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     target_pct.values,
...     np.asarray([100.]),  # (1)!
...     True,
...     False,
...     False
... )
```

1. Keyword arguments with default values must be provided explicitly

That was lightning fast! :zap:

!!! important
    You should ensure that the provided arguments exactly match the registered signature,
    otherwise you may get errors that may be very difficult to debug. For example, 
    while setting `init_cash` to `np.asarray(100.)` (without brackets) would yield an 
    _"index is out of bounds"_ error, casting the array to integer would make all allocations zero!

## Summary

We've covered in detail a lot of components of a typical simulator in vectorbt. Simulation
is the primary step in backtesting of a trading strategy, and by mastering it you'll
gain some hard-core skills that can be applied just in any place of the vectorbt's rich 
Numba ecosystem. 

One of the most important takeaways from this documentation piece is that implementing a custom 
simulator is as easy (or as difficult) as any other Numba-compiled function, and there is no point 
in using the preset simulation methods such as 
[Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
if you can produce the same results, achieve a multifold performance gain, be able to use rotational 
indexing, caching, and AOT compilation, by designing your own pipeline from scratch. After all, it's just 
a bunch of loops that gradually move over the shape of a matrix, execute orders, update the state 
of the simulation, and write some output data. Everything else is up to your imagination :mage: