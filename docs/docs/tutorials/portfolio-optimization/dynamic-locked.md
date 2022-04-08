---
title: Dynamic
---

# :material-lock-open: Portfolio optimization - Dynamic

Until now, all allocation and optimization functions were based strictly on external
information such as pricing data and had no control over the actual execution. But what 
if we want to rebalance based on some conditions within the current trading environment?
For instance, to perform a threshold rebalancing, we need to know the current portfolio value.
This scenario introduces a path-dependent problem, which can only be addressed using 
a custom order function.

Let's backtest threshold rebalancing, which is a portfolio management strategy used to maintain 
a set of desired allocations, without allowing the asset weightings from deviating excessively.
We'll create a template pipeline that takes any user-defined, Numba-compiled allocation function.
When one of the individual constituents of the portfolio crosses outside the bounds of their 
desired allocations, the entire portfolio is rebalanced to realign with the target allocations.

Here's a general template:

```pycon
>>> from vectorbtpro.portfolio.enums import SizeType, Direction
>>> from vectorbtpro.portfolio import nb as pf_nb
>>> from collections import namedtuple

>>> GroupMemory = namedtuple("GroupMemory", [  # (1)!
...     "target_alloc", 
...     "size_type",
...     "direction",
...     "order_value_out"
... ])

>>> @njit
... def pre_group_func_nb(c):  # (2)!
...     group_memory = GroupMemory(
...         target_alloc=np.full(c.group_len, np.nan),  # (3)!
...         size_type=np.full(c.group_len, SizeType.TargetPercent),  # (4)!
...         direction=np.full(c.group_len, Direction.Both),
...         order_value_out=np.full(c.group_len, np.nan)  # (5)!
...     )
...     return group_memory,

>>> @njit
... def pre_segment_func_nb(  # (6)!
...     c, 
...     group_memory,  # (7)!
...     min_history,  # (8)!
...     threshold,  # (9)!
...     allocate_func_nb,  # (10)!
...     *args
... ):
...     should_rebalance = False
...     
...     if c.i >= min_history:
...         in_position = False
...         for col in range(c.from_col, c.to_col):
...             if c.last_position[col] != 0:
...                 in_position = True
...                 break
...                 
...         if not in_position:
...             should_rebalance = True
...         else:
...             curr_value = c.last_value[c.group]
...             for group_col in range(c.group_len):
...                 col = c.from_col + group_col
...                 curr_position = c.last_position[col]
...                 curr_price = c.last_val_price[col]
...                 curr_alloc = curr_position * curr_price / curr_value
...                 curr_threshold = pf_nb.get_col_elem_nb(c, col, threshold)
...                 alloc_diff = curr_alloc - group_memory.target_alloc[group_col]
...                 
...                 if abs(alloc_diff) >= curr_threshold:
...                     should_rebalance = True
...                     break
...                     
...     if should_rebalance:
...         allocate_func_nb(c, group_memory, *args)  # (11)!
...         pf_nb.sort_call_seq_nb(  # (12)!
...             c, 
...             group_memory.target_alloc, 
...             group_memory.size_type, 
...             group_memory.direction, 
...             group_memory.order_value_out
...         )
...         
...     return group_memory, should_rebalance

>>> @njit
... def order_func_nb(  # (13)!
...     c, 
...     group_memory,  # (14)! 
...     should_rebalance, 
...     price,
...     fees
... ):
...     if not should_rebalance:
...         return pf_nb.order_nothing_nb()
...     
...     group_col = c.col - c.from_col  # (15)!
...     return pf_nb.order_nb(
...         size=group_memory.target_alloc[group_col], 
...         price=pf_nb.get_elem_nb(c, price),
...         size_type=group_memory.size_type[group_col],
...         direction=group_memory.direction[group_col],
...         fees=pf_nb.get_elem_nb(c, fees)
...     )
```

1. Create a named tuple acting as a container for variables that should be shared between contexts
within each portfolio group
2. Initialize arrays in each portfolio group
3. Array with target allocation per asset. Will be re-filled at each rebalancing step.
4. Arrays with size type and direction per asset. Defaults to target percentage and both directions, 
but you are free to change them from within `allocate_func_nb`.
5. Temporary array required for sorting by order value
6. Rebalancing should take place in a pre-segment function. Remember that segment is a group
of assets at a specific timestamp, and in this function we should decide whether to rebalance the entire group.
7. Arguments returned by `pre_group_func_nb` are passed right after the context
8. When to place the first allocation?
9. Threshold array, can be provided as a constant, or per timestamp, asset, or element
10. Allocation function that should take the context, the group memory, and `*args`
11. Allocation function should change arrays in the group context and return nothing (None)
12. Sort the current call sequence by order value to execute sell orders before buy orders
(to release funds early) using [sort_call_seq_nb](/api/portfolio/nb/from_order_func/#vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_nb)
13. For each asset in the current group (sorted by order value), vectorbt will call this order function
14. Again, the arguments returned by `pre_segment_func_nb` are passed right after the context
15. Index of the current column within the current group

Let's create an allocation function for an equally-weighted portfolio:

```pycon
>>> @njit
... def uniform_allocate_func_nb(c, group_memory):
...     for group_col in range(c.group_len):
...         group_memory.target_alloc[group_col] = 1 / c.group_len  # (1)!
```

1. Modify the target allocation array in-place

!!! hint
    Sometimes, we may want to rebalance dynamically based on a function that uses a third-party 
    library, such as SciPy or scikit-learn, and cannot be compiled with Numba. In such cases, 
    we can disable jitting of the main simulator function by passing `jitted=False`.

Now it's time to run the simulation!

```pycon
>>> def simulate_threshold_rebalancing(threshold, allocate_func_nb, *args, **kwargs):
...     return vbt.Portfolio.from_order_func(
...         data.get("Close"),
...         order_func_nb, 
...         vbt.Rep('price'),  # (1)!
...         vbt.Rep('fees'),
...         open=data.get("Open"),  # (2)!
...         pre_group_func_nb=pre_group_func_nb, 
...         pre_group_args=(),
...         pre_segment_func_nb=pre_segment_func_nb, 
...         pre_segment_args=(
...             0,  # (3)!
...             vbt.Rep("threshold"),  # (4)!
...             allocate_func_nb,
...             *args
...         ),
...         broadcast_named_args=dict(
...             price=data.get("Close"),
...             fees=0.005,
...             threshold=threshold
...         ),
...         cash_sharing=True,
...         group_by=vbt.ExceptLevel("symbol"),  # (5)!
...         freq='1h', 
...         **kwargs
...     )

>>> pf = simulate_threshold_rebalancing(0.05, uniform_allocate_func_nb)
>>> sim_alloc = pf.get_asset_value(group_by=False).vbt / pf.value
>>> sim_alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... )
```

1. Use templates to broadcast price together with `close`. Make sure to list such arguments
in `broadcast_named_args`.
2. Provide `open` to have a non-NA valuation price in `pre_segment_func_nb` at the first timestamp
3. We should place an allocation at the first timestamp
4. Threshold is an array-like object that broadcasts per timestamp and column,
thus broadcast price together with `close`
5. Group by everything except assets such that each group contains only assets

![](/assets/images/tutorials/pf_opt_dynamic.svg)

We see that threshold rebalancing makes asset allocations to repeatedly jump to their target levels. 

!!! info
    In cases where your kernel dies, or you want to validate the pipeline you created with Numba,
    it's advisable to either enable bound checks, or disable Numba entirely and then run your pipeline
    on sample data. This will effectively expose your hidden indexing bugs.

    For this, run the following in the first cell before anything else:

    ```pycon
    >>> import os
    
    >>> os.environ["NUMBA_BOUNDSCHECK"] = "1"
    >>> os.environ["NUMBA_DISABLE_JIT"] = "1"
    ```

We can also test multiple thresholds by simply making it an index:

```pycon
>>> pf = simulate_threshold_rebalancing(
...     pd.Index(np.arange(1, 16) / 100, name="threshold"),  # (1)!
...     uniform_allocate_func_nb
... )

>>> pf.sharpe_ratio
threshold
0.01         1.939476
0.02         1.964376
0.03         1.985139
0.04         1.984409
0.05         1.993303
0.06         2.019515
0.07         1.983011
0.08         2.047519
0.09         2.087105
0.10         1.939005
0.11         1.962413
0.12         1.978247
0.13         1.963005
0.14         1.969650
0.15         1.983107
Name: sharpe_ratio, dtype: float64
```

1. If we used `np.arange(0.01, 0.16, 0.01)`, we wouldn't be able to select those
values from columns since, for example, `0.6` would become `0.060000000000000005`

## Post-analysis

But can we somehow get the rebalancing timestamps? Of course!

```pycon
>>> @njit
... def track_uniform_allocate_func_nb(c, group_memory, index_points, alloc_counter):
...     for group_col in range(c.group_len):
...         group_memory.target_alloc[group_col] = 1 / c.group_len
...
...     index_points[alloc_counter[0]] = c.i
...     alloc_counter[0] += 1

>>> index_points = np.empty(data.wrapper.shape[0], dtype=np.int_)  # (1)!
>>> alloc_counter = np.full(1, 0)  # (2)!
>>> pf = simulate_threshold_rebalancing(
...     0.05,
...     track_uniform_allocate_func_nb, 
...     index_points, 
...     alloc_counter
... )
>>> index_points = index_points[:alloc_counter[0]]  # (3)!

>>> data.wrapper.index[index_points]
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-02-02 04:00:00+00:00',
               '2020-03-07 15:00:00+00:00', '2020-05-28 18:00:00+00:00',
               '2020-06-03 16:00:00+00:00', '2020-07-07 13:00:00+00:00',
               '2020-08-14 17:00:00+00:00', '2020-09-09 01:00:00+00:00',
               '2020-11-05 13:00:00+00:00', '2020-11-21 14:00:00+00:00',
               '2020-11-24 00:00:00+00:00', '2020-12-22 17:00:00+00:00',
               '2020-12-23 11:00:00+00:00', '2020-12-28 23:00:00+00:00',
               '2020-12-29 16:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

1. Since we don't know the number of index points in advance, we should initialize
an array of the same size as we have timestamps
2. Allocation counter is an array with one element - `0`. Why array and not a constant?
Because we need to keep a reference to it.
3. After successful simulation, a part of our empty index point array has been filled, but another
part is still uninitialized (i.e., it contains garbage). Use the counter to select the filled entries.

What if we want to post-analyze both, index points and target allocations?
And how should we treat cases when there are multiple parameter combinations? 

Allocations can be saved to an array in the same way as index points. But as soon as there are 
multiple groups, we have two options: either run the entire pipeline in a loop (remember that 
vectorbt sometimes even encourages you to do that because you can use chunking),
or simply concatenate index points and target allocations of all groups into a single array
and track the group of each entry in that array. We can then construct an instance of
[PortfolioOptimizer](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer) 
to conveniently post-analyze the entire target allocation data!

We need to make a few adaptations though. First, [PortfolioOptimizer](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer) 
requires index points to be of type [AllocPoints](/api/portfolio/pfopt/records/#vectorbtpro.portfolio.pfopt.records.AllocPoints),
which, in turn, requires the underlying data to be a structured array of a complex data type 
[alloc_point_dt](https://vectorbt.pro/api/portfolio/enums/#vectorbtpro.portfolio.enums.alloc_point_dt).
Second, our counter will track count per group rather than globally. By taking the sum of it,
we can still derive the global count. For a better illustration, we'll also implement a new allocation 
function that generates weights randomly. Finally, if you aren't scared of complexity and want the most 
flexible thing possible, see the "Flexible" tab for the same pipeline but with templates and in-outputs :smirk:

=== "Preset"

    ```pycon
    >>> from vectorbtpro.portfolio.enums import alloc_point_dt
    
    >>> @njit
    ... def random_allocate_func_nb(
    ...     c, 
    ...     group_memory, 
    ...     alloc_points, 
    ...     allocations, 
    ...     alloc_counter
    ... ):
    ...     weights = np.random.uniform(0, 1, c.group_len)
    ...     group_memory.target_alloc[:] = weights / weights.sum()
    ...
    ...     group_count = alloc_counter[c.group]
    ...     count = alloc_counter.sum()  # (1)!
    ...     alloc_points["id"][count] = group_count  # (2)!
    ...     alloc_points["col"][count] = c.group  # (3)!
    ...     alloc_points["alloc_idx"][count] = c.i  # (4)!
    ...     allocations[count] = group_memory.target_alloc  # (5)!
    ...     alloc_counter[c.group] += 1
    
    >>> thresholds = pd.Index(np.arange(1, 16) / 100, name="threshold")
    >>> max_entries = data.wrapper.shape[0] * len(thresholds)  # (6)!
    >>> alloc_points = np.empty(max_entries, dtype=alloc_point_dt)  # (7)!
    >>> allocations = np.empty((max_entries, len(data.symbols)), dtype=np.float_)  # (8)!
    >>> alloc_counter = np.full(len(thresholds), 0)  # (9)!
    
    >>> pf = simulate_threshold_rebalancing(
    ...     thresholds,
    ...     random_allocate_func_nb, 
    ...     alloc_points, 
    ...     allocations,
    ...     alloc_counter,
    ...     seed=42  # (10)!
    ... )
    >>> alloc_points = alloc_points[:alloc_counter.sum()]  # (11)!
    >>> allocations = allocations[:alloc_counter.sum()]
    ```

    1. Global count will be used as an index at which the current allocation data will be stored.
    This way, we incrementally write the allocation data of all groups into a single array.
    2. Record id is always per column (in our case per group since columns are groups in allocation records)
    3. Store the current column (in our case group since columns are groups in allocation records)
    4. Store the actual index point
    5. Store the allocation (vector of weights of group length)
    6. Since we cannot know the number of generated allocations in advance, we should prepare for a
    worst case where there is an allocation at each single timestamp, which yields the number of possible
    entries being the number of timestamps multiplied by the number of groups
    7. Notice how we use [alloc_point_dt](https://vectorbt.pro/api/portfolio/enums/#vectorbtpro.portfolio.enums.alloc_point_dt)
    as the data type of an array to make it structured
    8. Allocation array has the same number of entries as the index point array, but it has two dimensions
    because it must store the weight of each asset in a group (number of columns = number of assets in a group)
    9. Count is now stored per group
    10. Set seed to produce the same output every time we execute this cell
    11. To remove the remaining entires that haven't been filled, we need to use the global count,
    which is simply the sum of all group counts

=== "Flexible"

    ```pycon
    >>> from vectorbtpro.portfolio.enums import alloc_point_dt
    
    >>> @njit
    ... def random_allocate_func_nb(c, group_memory):  # (1)!
    ...     weights = np.random.uniform(0, 1, c.group_len)
    ...     group_memory.target_alloc[:] = weights / weights.sum()
    ...
    ...     group_count = c.in_outputs.alloc_counter[c.group]
    ...     count = c.in_outputs.alloc_counter.sum()
    ...     c.in_outputs.alloc_points["id"][count] = group_count
    ...     c.in_outputs.alloc_points["col"][count] = c.group
    ...     c.in_outputs.alloc_points["alloc_idx"][count] = c.i
    ...     c.in_outputs.allocations[count] = group_memory.target_alloc
    ...     c.in_outputs.alloc_counter[c.group] += 1
    
    >>> alloc_points = vbt.RepEval("""
    ...     max_entries = target_shape[0] * len(group_lens)
    ...     np.empty(max_entries, dtype=alloc_point_dt)
    ... """, context=dict(alloc_point_dt=alloc_point_dt))  # (2)!
    >>> allocations = vbt.RepEval("""
    ...     max_entries = target_shape[0] * len(group_lens)
    ...     np.empty((max_entries, n_cols), dtype=np.float_)
    ... """, context=dict(n_cols=len(data.symbols)))
    >>> alloc_counter = vbt.RepEval("np.full(len(group_lens), 0)")
    
    >>> InOutputs = namedtuple("InOutputs", [
    ...     "alloc_points",
    ...     "allocations",
    ...     "alloc_counter"
    ... ])
    >>> in_outputs = InOutputs(
    ...     alloc_points=alloc_points, 
    ...     allocations=allocations,
    ...     alloc_counter=alloc_counter,
    ... )
    
    >>> pf = simulate_threshold_rebalancing(
    ...     pd.Index(np.arange(1, 16) / 100, name="threshold"),  # (3)!
    ...     random_allocate_func_nb, 
    ...     in_outputs=in_outputs,  # (4)!
    ...     seed=42
    ... )
    >>> alloc_points = pf.in_outputs.alloc_points[:pf.in_outputs.alloc_counter.sum()]
    >>> allocations = pf.in_outputs.allocations[:pf.in_outputs.alloc_counter.sum()]
    ```

    1. The function is the same as in the "Preset" example, but the arrays are now provided
    via the in-outputs tuple rather than via arguments
    2. The same logic as in the "Preset" example, but we use templates to postpone the creation
    of all arrays to the point where all other arrays are broadcast and the final shape and
    the number of groups are available
    3. Notice how array creation isn't tied to the threshold array anymore!
    4. Since we now use templates, we don't have references to the created arrays anymore.
    But luckily, we can use in-outputs, which store the references to all arrays for us.

!!! hint
    If you perform portfolio optimization on some history of data (for example, by searching for the 
    maximum Sharpe ratio), make sure to use [alloc_range_dt](https://vectorbt.pro/api/portfolio/enums/#vectorbtpro.portfolio.enums.alloc_range_dt)
    and [AllocRanges](/api/portfolio/pfopt/records/#vectorbtpro.portfolio.pfopt.records.AllocRanges) -
    this would open another dimension in data analysis.

What's left is the creation of a [PortfolioOptimizer](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer) 
instance using the target allocation data that we just filled:

```pycon
>>> pf_opt = vbt.PortfolioOptimizer(  # (1)!
...     wrapper=pf.wrapper,  # (2)!
...     alloc_records=vbt.AllocPoints(
...         pf.wrapper.resolve(),  # (3)!
...         alloc_points
...     ),
...     allocations=allocations
... )
```

1. Notice how we don't call any class method - we have all the required data to instantiate the class directly!
2. Main wrapper should contain both, regular columns and groups
3. Allocation wrapper should contain only groups since the field `col` we filled previously
points to groups rather than regular columns. By using [ArrayWrapper.resolve](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.resolve), 
we can create a new wrapper where columns have been replaced by groups.

Having such an instance allows us to post-analyze the target allocation data.
Even though we used random weights in rebalancing, let's describe the allocations generated 
for the threshold of 10% just for the sake of example:

```pycon
>>> pf_opt[0.1].allocations.describe()
symbol   ADAUSDT   BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
count   6.000000  6.000000  6.000000  6.000000  6.000000
mean    0.159883  0.149608  0.156493  0.292615  0.241400
std     0.092490  0.079783  0.043584  0.098891  0.083152
min     0.076678  0.056292  0.094375  0.200873  0.098709
25%     0.091023  0.082134  0.149385  0.220957  0.223424
50%     0.123982  0.157974  0.153985  0.252078  0.243109
75%     0.230589  0.204527  0.156810  0.375171  0.293097
max     0.288493  0.248507  0.231013  0.423879  0.336853

>>> pf_opt.plot(column=0.1)
```

![](/assets/images/tutorials/pf_opt_dynamic_01.svg)

Here's how the target allocation picture changes with a lower threshold:

```pycon
>>> pf_opt.plot(column=0.03)
```

![](/assets/images/tutorials/pf_opt_dynamic_003.svg)

And here's what actually happened:

```pycon
>>> sim_alloc = pf[0.03].get_asset_value(group_by=False).vbt / pf[0.03].value
>>> sim_alloc.columns = data.symbols
>>> sim_alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... )
```

![](/assets/images/tutorials/pf_opt_dynamic_003_sim.svg)

Want the cool part? If we feed our manually-constructed optimizer instance to 
[Portfolio.from_optimizer](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_optimizer),
we'll get the exact same results :exploding_head:

```pycon
>>> pf.sharpe_ratio
threshold
0.01    1.098589
0.02    1.707435
0.03    1.774934
0.04    2.077395
0.05    2.082810
0.06    1.964397
0.07    2.106276
0.08    2.121431
0.09    1.838085
0.10    2.072324
0.11    2.228911
0.12    1.766233
0.13    1.859528
0.14    2.209064
0.15    2.124389
Name: sharpe_ratio, dtype: float64

>>> pf_new = vbt.Portfolio.from_optimizer(
...     pf_opt, 
...     data.get("Close"), 
...     val_price=data.get("Open"), 
...     freq="1h", 
...     fees=0.005
... )

>>> pf_new.sharpe_ratio
threshold
0.01    1.098589
0.02    1.707435
0.03    1.774934
0.04    2.077395
0.05    2.082810
0.06    1.964397
0.07    2.106276
0.08    2.121431
0.09    1.838085
0.10    2.072324
0.11    2.228911
0.12    1.766233
0.13    1.859528
0.14    2.209064
0.15    2.124389
Name: sharpe_ratio, dtype: float64
```

This proves once again how powerful vectorbt is: we just performed dynamic threshold rebalancing,
extracted the target allocation data from within the simulation, analyzed that data after the
simulation, and fed it to another, totally-different simulation method to make sure that we did 
no mistakes related to order generation.

## Bonus 1: Own optimizer

As a bonus, let's do a periodic mean-variance optimization using our own simulator! We'll 
generate the rebalancing dates in advance, and for each of them, we'll generate a bunch of Sharpe 
ratios for that period and use the Efficient Frontier to select the best one. The pipeline 
below is the most lightweight pipeline we can get: it processes only one parameter combination 
at a time using the vectorbt's low-level order execution API, and consumes only the information 
it really needs.

Here's our raw Numba-compiled pipeline (optimization function agnostic):

```pycon
>>> from vectorbtpro.portfolio.enums import (
...     ExecState, 
...     PriceArea, 
...     order_dt
... )
>>> from vectorbtpro.base.indexing import flex_select_auto_nb
>>> from vectorbtpro.generic import nb as generic_nb

>>> @njit(nogil=True)
... def optimize_portfolio_nb(
...     close, 
...     val_price,
...     index_ranges,
...     optimize_func_nb,
...     optimize_args=(),
...     price=np.asarray(np.inf),  # (1)!
...     fees=np.asarray(0.),
...     init_cash=100.,
...     group=0
... ):
...     order_records = np.empty(close.shape, dtype=order_dt)  # (2)!
...     order_counts = np.full(close.shape[1], 0, dtype=np.int_)
...     
...     order_value = np.empty(close.shape[1], dtype=np.float_)  # (3)!
...     call_seq = np.empty(close.shape[1], dtype=np.int_)
...     
...     last_position = np.full(close.shape[1], 0.0, dtype=np.float_)
...     last_debt = np.full(close.shape[1], 0.0, dtype=np.float_)
...     cash_now = float(init_cash)  # (4)!
...     free_cash_now = float(init_cash)
...     value_now = float(init_cash)
... 
...     for k in range(len(index_ranges)):  # (5)!
...         i = index_ranges[k][1]  # (6)!
...         size = optimize_func_nb(  # (7)!
...             index_ranges[k][0], 
...             index_ranges[k][1], 
...             *optimize_args
...         )
...         
...         # (8)!
...         value_now = cash_now
...         for col in range(close.shape[1]):
...             val_price_now = flex_select_auto_nb(val_price, i, col, True)
...             value_now += last_position[col] * val_price_now
...         
...         for col in range(close.shape[1]):
...             val_price_now = flex_select_auto_nb(val_price, i, col, True)
...             exec_state = ExecState(
...                 cash=cash_now,
...                 position=last_position[col],
...                 debt=last_debt[col],
...                 free_cash=free_cash_now,
...                 val_price=val_price_now,
...                 value=value_now,
...             )
...             order_value[col] = pf_nb.approx_order_value_nb(  # (9)!
...                 exec_state,
...                 size[col],
...                 SizeType.TargetPercent,
...                 Direction.Both,
...             )
...             call_seq[col] = col  # (10)!
... 
...         pf_nb.insert_argsort_nb(order_value, call_seq)  # (11)!
... 
...         for c in range(close.shape[1]):  # (12)!
...             col = call_seq[c]  # (13)!
...             
...             order = pf_nb.order_nb(  # (14)!
...                 size=size[col],
...                 price=flex_select_auto_nb(price, i, col, True),
...                 size_type=SizeType.TargetPercent,
...                 direction=Direction.Both,
...                 fees=flex_select_auto_nb(fees, i, col, True),
...             )
...
...             # (15)!
...             price_area = PriceArea(
...                 open=np.nan,
...                 high=np.nan,
...                 low=np.nan,
...                 close=flex_select_auto_nb(close, i, col, True),
...             )
...             val_price_now = flex_select_auto_nb(val_price, i, col, True)
...             exec_state = ExecState(
...                 cash=cash_now,
...                 position=last_position[col],
...                 debt=last_debt[col],
...                 free_cash=free_cash_now,
...                 val_price=val_price_now,
...                 value=value_now,
...             )
...             new_exec_state, order_result = pf_nb.process_order_nb(
...                 group=group,
...                 col=col,
...                 i=i,
...                 exec_state=exec_state,
...                 order=order,
...                 price_area=price_area,
...                 order_records=order_records,
...                 order_counts=order_counts
...             )
... 
...             cash_now = new_exec_state.cash
...             free_cash_now = new_exec_state.free_cash
...             value_now = new_exec_state.value
...             last_position[col] = new_exec_state.position
...             last_debt[col] = new_exec_state.debt
... 
...     # (16)!
...     return generic_nb.repartition_nb(order_records, order_counts)
```

1. Any array-like object can be provided as a constant, or per timestamp, asset, or element.
Just make sure that it's of type `np.ndarray`.
2. Since we don't know the number of orders in advance, create an array with the same number
of records as there are elements in `close`. Also note that `order_records` must be two-dimensional.
3. Since we're iterating over timestamps and columns, we need to somehow store information 
related to each asset. We do this using a one-dimensional array with elements aligned per column.
4. Every information associated with cash is a constant since we have only one group with cash sharing.
We'll update those constants at each iteration.
5. There is no significant difference in iteration over the entire shape of `close` or only 
over the timestamps where the optimization should really take place, thus we iterate over `index_ranges`.
If you additionally want to pair the optimization with a stop loss or other continuous checks,
you should iterate over `close.shape[0]` and run `optimize_func_nb` only if the current iteration
can be found in the second column of `index_ranges`.
6. Allocation index (at which optimization and order execution should take place) is by default 
the right bound in `index_ranges`
7. Run the optimization function, which should return the weights. You can adapt it to return
the size type, direction, and other information as well.
8. Calculate the current group value using the open price
9. Using the group value, approximate the order value of each allocation
10. Prepare the call sequence array
11. Sort the call sequence by order value, such that assets that should be sold are processed first
12. Iterate over each asset to place an order
13. Get the column index of an asset based on the sorted call sequence. For example,
if the first element in `call_seq` is 2, we should execute the third asset first.
14. Create an order. You can forward additional information such as fixed fees to this function.
15. Process the order and update the current balances
16. Order records are stored per column -> flatten them

And here's our Numba-compiled MVO function:

```pycon
>>> @njit(nogil=True)  # (1)!
... def sharpe_optimize_func_nb(
...     start_idx, 
...     end_idx, 
...     close,  # (2)!
...     num_tests, 
...     ann_factor
... ):
...     close_period = close[start_idx:end_idx]  # (3)!
...     returns = (close_period[1:] - close_period[:-1]) / close_period[:-1]
...     mean = generic_nb.nanmean_nb(returns)
...     cov = np.cov(returns, rowvar=False)
...     best_sharpe_ratio = -np.inf
...     weights = np.full(close.shape[1], np.nan, dtype=np.float_)
...     
...     for i in range(num_tests):  # (4)!
...         w = np.random.random_sample(close.shape[1])
...         w = w / np.sum(w)
...         p_return = np.sum(mean * w) * ann_factor
...         p_std = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(ann_factor)
...         sharpe_ratio = p_return / p_std
...         if sharpe_ratio > best_sharpe_ratio:
...             best_sharpe_ratio = sharpe_ratio
...             weights = w
...             
...     return weights
```

1. Remember to disable GIL everywhere to be able to use multi-threading
2. Those are additional arguments passed via `*args` in `optimize_portfolio_nb`
3. Select the previous week in `close`
4. Iterate over a number of tests, and for each, generate a random allocation, 
calculate its Sharpe ratio, and store it if it's better than other Sharpe ratios

Let's run the MVO on a weekly basis:

```pycon
>>> index_ranges = data.wrapper.get_index_ranges(every="W")
>>> ann_factor = pd.Timedelta("365d") / pd.Timedelta("1h")
>>> init_cash = 100
>>> num_tests = 30
>>> fees = np.asarray(0.005)

>>> order_records = optimize_portfolio_nb(
...     data.get("Close").values,
...     data.get("Open").values,
...     index_ranges,
...     sharpe_optimize_func_nb,
...     optimize_args=(data.get("Close").values, num_tests, ann_factor),
...     fees=fees,
...     init_cash=init_cash
... )
```

The result of our optimization are order records, which can be used as an input
to the [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) class:

```pycon
>>> pf = vbt.Portfolio(
...     wrapper=symbol_wrapper.regroup(True),  # (1)!
...     close=data.get("Close"), 
...     order_records=order_records, 
...     log_records=np.array([]),  # (2)!
...     cash_sharing=True, 
...     init_cash=init_cash
... )
```

1. By using [ArrayWrapper.regroup](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.regroup), 
we can make the wrapper grouped (`True` means putting all assets into one group)
2. We have no log records. If you need logs, you can generate and process them in the same way
as order records.

We can now analyze the portfolio as we usually do!

```pycon
>>> sim_alloc = pf.get_asset_value(group_by=False).vbt / pf.value
>>> sim_alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... )
```

![](/assets/images/tutorials/pf_opt_mvo.svg)

In contrast to all the examples above, our pipeline can process only one parameter combination
at once. This means that we should test all the parameter combinations in a loop, which can be
easily done either manually, or by using the vectorbt's execution function - 
[execute](/api/utils/execution/#vectorbtpro.utils.execution.execute) - allowing us to distribute 
the execution, for example, with Dask. The function requires us providing a sequence of functions 
and their arguments, one combination per execution. 

Let's test the Cartesian product of different index ranges, number of tests, and fees:

```pycon
>>> test_index_ranges = pd.Index(["D", "W", "MS"], name="every")  # (1)!
>>> test_num_tests = pd.Index([30, 50, 100], name="num_tests")
>>> test_fees = pd.Index([0.0, 0.005, 0.01], name="fees")

>>> index_ranges_cache = {}  # (2)!
>>> for every in test_index_ranges:
...     index_ranges_cache[every] = symbol_wrapper.get_index_ranges(every=every)

>>> param_index = vbt.combine_indexes((  # (3)!
...     test_index_ranges, 
...     test_num_tests,
...     test_fees
... ))

>>> kwargs = dict(  # (4)!
...     close=data.get("Close").values,
...     val_price=data.get("Open").values,
...     optimize_func_nb=sharpe_optimize_func_nb,
...     init_cash=init_cash
... )
>>> funcs_args = []
>>> for i in range(len(param_index)):  # (5)!
...     funcs_args.append((
...         optimize_portfolio_nb,
...         (),
...         {
...             **kwargs,
...             "index_ranges": index_ranges_cache[param_index[i][0]],
...             "optimize_args": (
...                 kwargs["close"],
...                 param_index[i][1],
...                 ann_factor
...             ),
...             "fees": np.asarray(param_index[i][2]),
...             "group": i
...         }
...     ))
>>> order_records_list = vbt.execute(  # (6)!
...     funcs_args, 
...     chunk_len=4, 
...     engine="dask"
... )
```

1. Create a Pandas Index instead of a list or other sequence to build a column hierarchy later
2. Index ranges should be calculated and cached prior to the execution
3. Combine all indexes using [combine_indexes](/api/base/indexes/#vectorbtpro.base.indexes.combine_indexes)
to form a single column hierarchy. This will build the Cartesian product of all index elements.
4. These keyword arguments do not depend on our parameters
5. Iterate over each parameter combination, and create keyword arguments that should
be passed to `optimize_portfolio_nb` as part of this combination
6. Build chunks of 4 function calls. Calls within each chunk are distributed with Dask,
while chunks themselves are processed sequentially.

[=100% "Chunk 7/7"]{: .candystripe}

Using Dask, each function call executes in roughly 10 milliseconds! 

After we've generated the order records for each parameter combination, let's iterate over 
this list once again to assess the stats:

```pycon
>>> sharpe_ratios = pd.Series(index=param_index, dtype=np.float_)
>>> for i, order_records in enumerate(order_records_list):
...     pf = vbt.Portfolio(
...         wrapper=symbol_wrapper.regroup(True), 
...         close=data.get("Close"), 
...         order_records=order_records, 
...         log_records=np.array([]), 
...         cash_sharing=True, 
...         init_cash=init_cash
...     )
...     sharpe_ratios[i] = pf.sharpe_ratio

>>> sharpe_ratios
every  num_tests  fees 
D      30         0.000    2.017888
                  0.005    0.523525
                  0.010   -1.382062
       50         0.000    2.148706
                  0.005    0.236855
                  0.010   -1.410316
       100        0.000    2.205136
                  0.005    0.260073
                  0.010   -1.567652
W      30         0.000    2.374737
                  0.005    1.889980
                  0.010    2.009748
       50         0.000    2.363708
                  0.005    2.171151
                  0.010    1.796394
       100        0.000    2.577051
                  0.005    2.271735
                  0.010    2.088430
MS     30         0.000    1.540923
                  0.005    1.399212
                  0.010    1.809829
       50         0.000    1.630768
                  0.005    1.808740
                  0.010    1.723790
       100        0.000    1.809452
                  0.005    1.521233
                  0.010    1.839857
dtype: float64
```

We could have also stacked all the order records and analyzed them as part of a single portfolio,
but this would require tiling the close price by the number of parameter combinations,
which could become memory-expensive very quickly, thus basic looping is preferred here.

## Bonus 2: Hyperopt

Instead of constructing and testing the full parameter grid, we can adopt a statistical approach.
There are libraries, such as Hyperopt, that are tailored at minimizing objective functions.

> [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for serial and parallel 
> optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.

To use Hyperopt, we need to implement the objective function first. This is an easy task in our case:

```pycon
>>> def objective(kwargs):
...     close_values = data.get("Close").values
...     open_values = data.get("Open").values
...     index_ranges = symbol_wrapper.get_index_ranges(every=kwargs["every"])
...     order_records = optimize_portfolio_nb(
...         close_values,
...         open_values,
...         index_ranges,
...         sharpe_optimize_func_nb,
...         optimize_args=(close_values, kwargs["num_tests"], ann_factor),
...         fees=np.asarray(kwargs["fees"]),
...         init_cash=init_cash
...     )
...     pf = vbt.Portfolio(
...         wrapper=symbol_wrapper.regroup(True), 
...         close=data.get("Close"), 
...         order_records=order_records, 
...         log_records=np.array([]), 
...         cash_sharing=True, 
...         init_cash=init_cash
...     )
...     return -pf.sharpe_ratio  # (1)!
```

1. Must be a float-valued function value that you are trying to **minimize**

Then, we need to construct the grid:

```pycon
>>> from hyperopt import fmin, tpe, hp

>>> space = {
...     "every": hp.choice("every", ["%dD" % n for n in range(1, 100)]),
...     "num_tests": hp.quniform("num_tests", 5, 100, 1),
...     "fees": hp.uniform('fees', 0, 0.05)
... }
```

Finally, let's search for the best candidate:

```pycon
>>> best = fmin(
...     fn=objective,
...     space=space,
...     algo=tpe.suggest,
...     max_evals=30
... )
100%|██████| 30/30 [00:01<00:00, 24.11trial/s, best loss: -2.4913128485273424]

>>> best
{'every': 94, 'fees': 0.018010147914768106, 'num_tests': 5.0}
```

Here's the [official tutorial](https://github.com/hyperopt/hyperopt/wiki/FMin) to help you get started.

## Summary

With regular portfolio review, we can make adjustments and increase the likelihood we'll end up 
with comfortable returns while maintaining the amount of risk we're willing to carry.
Diversification across asset classes is a risk-mitigation strategy, especially
when spreading investments across a variety of asset classes. With vectorbt, we have 
powerful functionalities at our disposal to select optimal portfolios in a programmatic way.
Not only there are tools that play well with third-party libraries, but there is an entire
universe of options to easily implement and test any unique optimization strategy,
especially when it takes advantage of acceleration, such as by compilation with Numba.

As we saw in this set of examples, vectorbt encourages us to adopt data science and to look at 
portfolio optimization from many angles to better understand how it affects the results. For instance, 
we can use the [PortfolioOptimizer](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer) 
class to quickly tune various parameters in weight generation and rebalancing timing, and 
upon a satisfactory pre-analysis, feed the optimizer into a simulator to post-analyze the chosen strategy.
Or, we can decide to implement our own optimizer from the ground up to control the
entire execution process. In this case, we can extract target allocations and other metadata
from within the simulation and analyze them later. So many possibilities... :thought_balloon:

[:material-lock: Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/locked-notebooks.md){ .md-button target="blank_" }
