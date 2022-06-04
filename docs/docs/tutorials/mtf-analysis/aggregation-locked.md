---
title: Aggregation
---

# :material-lock-open: MTF analysis - Aggregation

Aggregation plays a central role in downsampling. Consider a use case where we want to know
the maximum drawdown (MDD) of each month of data. Let's do this using various different techniques
available in vectorbt. The first approach involves resampling the data and then manipulating it:

```pycon
>>> ms_data = h1_data.resample("MS")
>>> ms_data.get("Low") / ms_data.get("High") - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

The same can be done by resampling only the arrays that are needed for the calculation:

```pycon
>>> h1_high = h1_data.get("High")
>>> h1_low = h1_data.get("Low")
>>> ms_high = h1_high.resample("MS").max()
>>> ms_low = h1_low.resample("MS").min()
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

And now using the vectorbt's superfast [GenericAccessor.resample_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_apply),
which uses Numba:

```pycon
>>> ms_high = h1_high.vbt.resample_apply("MS", vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.resample_apply("MS", vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

!!! hint
    See available reduce functions ending with `reduce_nb` in [nb.apply_reduce](/api/generic/nb/apply_reduce/).
    If you cannot find some function, you can always write it yourself :wink:

## Custom index

Using rules such as `MS` is very convenient but still not enough for many use cases. Consider a scenario
where we already have a target index we would want to resample to: none of Pandas functions allow
for such flexibility, unless we can somehow express the operation using 
[pandas.DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html).
Luckily, vectorbt allows for a variety of inputs and options to make this possible.

### Using target index

The method [GenericAccessor.resample_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_apply) 
has two different modes: the one that uses the target index (see [GenericAccessor.resample_to_index](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_to_index)), 
and the one that uses a Pandas resampler and vectorbt's grouping mechanism 
(see [GenericAccessor.groupby_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.groupby_apply)). 
The first one is the default mode: it's very fast but requires careful handling of bounds. 
The second one is guaranteed to produce the same results as Pandas but is (considerably) slower,
and can be enabled by passing `use_groupby_apply=True` to [GenericAccessor.resample_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_apply).

Talking about the first mode, it actually works in a similar fashion to [GenericAccessor.latest_at_index](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.latest_at_index) 
by taking the source and target index, and aggregating all the array elements located between each
two timestamps in the target index. This is done in one pass for best efficiency. And also similar to
`latest_at_index`, we can pass a [Resampler](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler)
instance and so provide our own custom index, even a numeric one. But in contrast to `latest_at_index`, 
there is no argument to specify frequencies or bounds - the left/right bound is always the previous/next element in 
the target index (or infinity). This is best illustrated in the following example:

```pycon
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> h1_high.vbt.resample_to_index(
...     target_index, 
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    29300.0
Name: High, dtype: float64
```

!!! info
    You should only think about this whenever passing a custom index. Passing a frequency like `MS`
    will produce results identical to that of Pandas with default arguments.

We see that the second value takes the maximum out of all values coming after `2020-02-01`,
which is not intended since we want the aggregation to be performed strictly per month.
To solve this, let's add another index value that will act as the rightmost bound:

```pycon
>>> target_rbound_index = vbt.Resampler.get_rbound_index(  # (1)!
...     target_index, 
...     pd.offsets.MonthBegin(1)
... )
>>> h1_high.vbt.resample_to_index(
...     target_index.append(target_rbound_index[[-1]]), 
...     vbt.nb.max_reduce_nb
... ).iloc[:-1]
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64

>>> h1_high[:"2020-03-01"].resample("MS").max().iloc[:-1]  # (2)!
Open time
2020-01-01 00:00:00+00:00     9578.0
2020-02-01 00:00:00+00:00    10500.0
Freq: MS, Name: High, dtype: float64
```

1. Get the right bound index for `target_index` using 
[Resampler.get_rbound_index](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler.get_rbound_index)
(class method)
2. Validate the output using Pandas

### Using group-by

The second mode has a completely different implementation: it creates or takes a 
[Pandas Resampler](https://pandas.pydata.org/docs/reference/resampling.html) or a
[Pandas Grouper](https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html), and parses 
them to build a [Grouper](/api/base/grouping/#vectorbtpro.base.grouping.base.Grouper) instance.
The grouper stores a map linking each group of elements in the source index to the respective elements 
in the target index. This map is then passed to a Numba-compiled function for aggregation per group.

Enough theory! Let's perform our resampling procedure using the grouping mechanism:

```pycon
>>> pd_resampler = h1_high.resample("MS")
>>> ms_high = h1_high.vbt.groupby_apply(pd_resampler, vbt.nb.max_reduce_nb)
>>> ms_low = h1_low.vbt.groupby_apply(pd_resampler, vbt.nb.min_reduce_nb)
>>> ms_low / ms_high - 1
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
dtype: float64
```

But since parsing a resampler or grouper object from Pandas is kinda slow, we can provide
our own grouper that can considerably speed the things up. Here we have two options:
either providing any `group_by` object, such as a Pandas Index, a NumPy array, or a level name
in a multi-index level, or a [Grouper](/api/base/grouping/#vectorbtpro.base.grouping.base.Grouper) instance itself.

Below, we will aggregate the data by month index:

```pycon
>>> h1_high.vbt.groupby_apply(h1_high.index.month, vbt.nb.max_reduce_nb)
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

Which is similar to calling [pandas.DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html):

```pycon
>>> h1_high.groupby(h1_high.index.month).max()
Open time
1      9578.00
2     10500.00
3      9188.00
4      9460.00
5     10067.00
6     10380.00
7     11444.00
8     12468.00
9     12050.85
10    14100.00
11    19863.16
12    29300.00
Name: High, dtype: float64
```

!!! hint
    Using built-in functions such as `max` when using Pandas resampling and grouping are already optimized
    and are on par with vectorbt regarding performance. Consider using vectorbt's functions mainly 
    when you have a custom function and you are forced to use `apply` - that's where vectorbt really shines :sunny:

### Using bounds

We've just learned that [GenericAccessor.resample_to_index](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_to_index)
aggregates all the array values that come after/before each element in the target index,
while [GenericAccessor.groupby_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.groupby_apply)
aggregates all the array values that map to the same target index by binning. But the first method 
doesn't allow gaps and custom bounds, while the second method doesn't allow overlapping groups. 
Both of these limitations are solved by [GenericAccessor.resample_between_bounds](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_between_bounds)!

This method takes the left and the right bound of the target index, and aggregates all the array values
that fall in between those two bounds:

```pycon
>>> target_lbound_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> target_rbound_index = pd.Index([
...     "2020-02-01",
...     "2020-03-01",
... ])
>>> h1_high.vbt.resample_between_bounds(  # (1)!
...     target_lbound_index, 
...     target_rbound_index,
...     vbt.nb.max_reduce_nb
... )
2020-01-01     9578.0
2020-02-01    10500.0
Name: High, dtype: float64
```

1. By default, the left bound is closed (included in the aggregation) and the right bound is open
(excluded from the aggregation). To change that, use `closed_lbound` and `closed_rbound`.

This opens some very interesting possibilities, such as custom-sized expanding windows. Let's calculate
the highest high up to the beginning of each month:

```pycon
>>> h1_high.vbt.resample_between_bounds(
...     "2020-01-01", 
...     pd.date_range("2020-01-02", "2021-01-01", freq="MS"),
...     vbt.nb.max_reduce_nb
... )  # (1)!
2020-02-01     9578.00
2020-03-01    10500.00
2020-04-01    10500.00
2020-05-01    10500.00
2020-06-01    10500.00
2020-07-01    10500.00
2020-08-01    11444.00
2020-09-01    12468.00
2020-10-01    12468.00
2020-11-01    14100.00
2020-12-01    19863.16
2021-01-01    29300.00
Freq: MS, Name: High, dtype: float64
```

1. The returned index contains the highest high before (and not including) each of the dates

Let's validate the output:

```pycon
>>> h1_high.expanding().max().resample("MS").max()
Open time
2020-01-01 00:00:00+00:00     9578.00
2020-02-01 00:00:00+00:00    10500.00
2020-03-01 00:00:00+00:00    10500.00
2020-04-01 00:00:00+00:00    10500.00
2020-05-01 00:00:00+00:00    10500.00
2020-06-01 00:00:00+00:00    10500.00
2020-07-01 00:00:00+00:00    11444.00
2020-08-01 00:00:00+00:00    12468.00
2020-09-01 00:00:00+00:00    12468.00
2020-10-01 00:00:00+00:00    14100.00
2020-11-01 00:00:00+00:00    19863.16
2020-12-01 00:00:00+00:00    29300.00
Freq: MS, Name: High, dtype: float64
```

## Meta methods

All the methods introduced above are great when the primary operation should be performed on **one** array.
But as soon as the operation involves multiple arrays (like `h1_high` and `h1_low` in our example),
we need to perform multiple resampling operations and make sure that the results align nicely. 
A cleaner approach would be to do a resampling operation that does the entire calculation
in one single pass, which is best for performance and consistency. Such operations can be performed
using meta methods. 

Meta methods are class methods that aren't bound to any particular array and that can take, 
broadcast, and combine more than one array of data. And the good thing is: most of the methods 
we used above are also available as meta methods! Let's calculate the MDD using a single resampling 
operation with [GenericAccessor.resample_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_apply):

```pycon
>>> from numba import njit

>>> @njit  # (1)!
... def mdd_nb(from_i, to_i, col, high, low):  # (2)!
...     highest = np.nanmax(high[from_i:to_i, col])  # (3)!
...     lowest = np.nanmin(low[from_i:to_i, col])
...     return lowest / highest - 1  # (4)!

>>> vbt.pd_acc.resample_apply(  # (5)!
...     'MS',
...     mdd_nb,
...     vbt.Rep('high'),  # (6)!
...     vbt.Rep('low'),
...     broadcast_named_args=dict(  # (7)!
...         high=h1_high,
...         low=h1_low
...     )
... )
Open time
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

1. Write a regular Python function and decorate it with `@njit` to compile with Numba
2. Since a meta method isn't bound to any array, the calculation function must accept the 
*meta*data of each selection, which is the index range `from_i` and `to_i`, and the column `col`
3. Select the respective elements from each array using the supplied metadata. Here, `highest`
is the highest price registered in this particular month, which starts at index `from_i` (inclusive)
and ends at index `to_i` (exclusive)
4. The reduction function should always return a single element
5. `vbt.pd_acc.resample_apply` is an alias for `pd.DataFrame.vbt.resample_apply`. Notice that we now call
`resample_apply` as a **class** method.
6. We want to take the high and low price as additional arguments after the metadata arguments.
Pass them right after the reduction function as templates, which get resolved after broadcasting.
7. Putting any arrays to this dict will broadcast them to the same shape and convert them to 
two-dimensional NumPy arrays

You can think of meta methods as flexible siblings of regular methods: they act as micro-pipelines
that take an arbitrary number of arrays and allow us to select the elements of those array as we wish.
If we place a print statement in `mdd_nb` to print out `from_i`, `to_i`, and `col`, we would get:

```pycon
0 744 0
744 1434 0
1434 2177 0
2177 2895 0
2895 3639 0
3639 4356 0
4356 5100 0
5100 5844 0
5844 6564 0
6564 7308 0
7308 8027 0
8027 8767 0
```

Each of those lines is a separate `mdd_nb` call, while the first two indices in each line denote
the absolute start and end index we should select from data. Since we used `MS` as a target frequency, 
`from_i` and `to_i` denote the start and end of the month respectively. We can actually prove this:

```pycon
>>> h1_high.iloc[0:744]  # (1)!
Open time
2020-01-01 00:00:00+00:00    7196.25
2020-01-01 01:00:00+00:00    7230.00
2020-01-01 02:00:00+00:00    7244.87
...                              ...
2020-01-31 21:00:00+00:00    9373.85
2020-01-31 22:00:00+00:00    9430.00
2020-01-31 23:00:00+00:00    9419.96
Name: High, Length: 744, dtype: float64

>>> h1_low.iloc[0:744].min() / h1_high.iloc[0:744].max() - 1
-0.28262267696805177
```

1. The first pair of indices corresponds to January 2020

The same example using [GenericAccessor.resample_between_bounds](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_between_bounds):

```pycon
>>> target_lbound_index = pd.date_range("2020-01-01", "2020-12-01", freq="MS", tz="UTC")
>>> target_rbound_index = pd.date_range("2020-02-01", "2021-01-01", freq="MS", tz="UTC")
>>> vbt.pd_acc.resample_between_bounds(
...     target_lbound_index,
...     target_rbound_index,
...     mdd_nb,
...     vbt.Rep('high'),
...     vbt.Rep('low'),
...     broadcast_named_args=dict(
...         high=h1_high,
...         low=h1_low
...     )
... )
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

Sky is the limit when it comes to possibilities that vectorbt enables for analysis :milky_way:

## Numba

90% of functionality in vectorbt is compiled with Numba. To avoid using the high-level API and dive
deep into the world of Numba, just look up in the documentation the Numba-compiled function used by 
the accessor function you want to use. For example, [GenericAccessor.resample_between_bounds](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_between_bounds)
first generates index ranges using [map_bounds_to_source_ranges_nb](/api/base/resampling/nb/#vectorbtpro.base.resampling.nb.map_bounds_to_source_ranges_nb)
and then uses [reduce_index_ranges_nb](/api/generic/nb/resample/#vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_nb)
for generic calls and [reduce_index_ranges_meta_nb](/api/generic/nb/resample/#vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_meta_nb)
for meta calls. Let's run the same meta function as above:

```pycon
>>> from vectorbtpro.base.resampling.nb import map_bounds_to_source_ranges_nb

>>> range_starts, range_ends = map_bounds_to_source_ranges_nb(  # (1)!
...     source_index=h1_high.index.values,
...     target_lbound_index=target_lbound_index.values,
...     target_rbound_index=target_rbound_index.values,
...     closed_lbound=True,
...     closed_rbound=False,
... )
>>> np.column_stack((range_starts, range_ends))  # (2)!
array([[   0,  744],
       [ 744, 1434],
       [1434, 2177],
       [2177, 2895],
       [2895, 3639],
       [3639, 4356],
       [4356, 5100],
       [5100, 5844],
       [5844, 6564],
       [6564, 7308],
       [7308, 8027],
       [8027, 8767]])

>>> ms_mdd_arr = vbt.nb.reduce_index_ranges_meta_nb(  # (3)!
...     1,  # (4)!
...     range_starts,
...     range_ends,
...     mdd_nb,
...     vbt.to_2d_array(h1_high),  # (5)!
...     vbt.to_2d_array(h1_low)
... )
>>> ms_mdd_arr
array([[-0.28262268],
       [-0.19571429],
       [-0.58836199],
       [-0.34988266],
       [-0.1937022 ],
       [-0.14903661],
       [-0.22290895],
       [-0.15636028],
       [-0.18470481],
       [-0.26425532],
       [-0.33570238],
       [-0.40026177]])
```

1. The first function iterates over each pair of bounds and returns the start and end index
in the source index within that bounds
2. The returned indices are in fact the same indices we printed out previously.
Use `np.column_stack` to print start and end indices next to each other.
3. The second function iterates over each index range and calls `mdd_nb` on it
4. Number of columns in `h1_high` and `h1_low`
5. Custom arrays should be two-dimensional

That's the fastest execution we can get. We can then wrap the array as follows:

```pycon
>>> pd.Series(ms_mdd_arr[:, 0], index=target_lbound_index)
2020-01-01 00:00:00+00:00   -0.282623
2020-02-01 00:00:00+00:00   -0.195714
2020-03-01 00:00:00+00:00   -0.588362
2020-04-01 00:00:00+00:00   -0.349883
2020-05-01 00:00:00+00:00   -0.193702
2020-06-01 00:00:00+00:00   -0.149037
2020-07-01 00:00:00+00:00   -0.222909
2020-08-01 00:00:00+00:00   -0.156360
2020-09-01 00:00:00+00:00   -0.184705
2020-10-01 00:00:00+00:00   -0.264255
2020-11-01 00:00:00+00:00   -0.335702
2020-12-01 00:00:00+00:00   -0.400262
Freq: MS, dtype: float64
```

## Caveats

As we already discussed in [Alignment](#alignment), each timestamp is the open time and information
at that timestamp happens somewhere between this timestamp and the next one. We shouldn't worry
about this if we downsample to a frequency that is an integer multiplier of the source frequency.
For example, consider downsampling two days of `H4` data to `D1` time frame:

```pycon
>>> h4_close_2d = h4_close.iloc[:12]
>>> h4_close_2d
Open time
2020-01-01 00:00:00+00:00    7225.01
2020-01-01 04:00:00+00:00    7209.83
2020-01-01 08:00:00+00:00    7197.20
2020-01-01 12:00:00+00:00    7234.19
2020-01-01 16:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    7129.61
2020-01-02 04:00:00+00:00    7110.57
2020-01-02 08:00:00+00:00    7139.79
2020-01-02 12:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6983.27
2020-01-02 20:00:00+00:00    6965.71
Freq: 4H, Name: Close, dtype: float64

>>> h4_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    6965.71
Freq: D, Name: Close, dtype: float64
```

This operation is correct: `7200.85` is the last value of `2020-01-01` and
`6965.71` is the last value of `2020-01-02`. But what happens if we change `H4` to `H5`? Nothing good:

```pycon
>>> h5_close = h1_close.resample("5h").last()
>>> h5_close_2d = h5_close.iloc[:10]
>>> h5_close_2d
Open time
2020-01-01 00:00:00+00:00    7217.27
2020-01-01 05:00:00+00:00    7188.77
2020-01-01 10:00:00+00:00    7221.43
2020-01-01 15:00:00+00:00    7229.48
2020-01-01 20:00:00+00:00    7211.02
2020-01-02 01:00:00+00:00    7138.93
2020-01-02 06:00:00+00:00    7161.83
2020-01-02 11:00:00+00:00    7130.98
2020-01-02 16:00:00+00:00    6948.49
2020-01-02 21:00:00+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64

>>> h5_close_2d.resample("1d").last()
Open time
2020-01-01 00:00:00+00:00    7211.02
2020-01-02 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

Try spotting the issue and come back once you found it (or not)...

Pandas resampler thinks that information at each timestamp happens exactly at that timestamp,
and so it chose the latest value of the first day to be at the latest timestamp of that day - 
`2020-01-01 20:00:00`. But this is a no-go for us! The timestamp `2020-01-01 20:00:00` holds the 
close price, which happens right before the next timestamp, or `2020-01-02 01:00:00` on the next day. 
This value is still unavailable at the end of the first day. Using this information that early
means looking into the future, and producing unreliable backtesting results.

This happens only when the target frequency cannot be divided by the source frequency without a leftover:

```pycon
>>> pd.Timedelta("1d") % pd.Timedelta("1h")  # (1)!
Timedelta('0 days 00:00:00')

>>> pd.Timedelta("1d") % pd.Timedelta("4h")  # (2)!
Timedelta('0 days 00:00:00')

>>> pd.Timedelta("1d") % pd.Timedelta("5h")  # (3)!
Timedelta('0 days 04:00:00')
```

1. Won't cause issues
2. Won't cause issues
3. Needs adjustment

But the solution is rather simple: make each timestamp be the close time instead of the open time.
Logically, the close time is just the next timestamp minus one nanosecond (the smallest timedelta possible):

```pycon
>>> h5_close_time = h5_close_2d.index.shift("5h") - pd.Timedelta(nanoseconds=1)
>>> h5_close_time.name = "Close time"
>>> h5_close_2d.index = h5_close_time
>>> h5_close_2d
Close time
2020-01-01 04:59:59.999999999+00:00    7217.27
2020-01-01 09:59:59.999999999+00:00    7188.77
2020-01-01 14:59:59.999999999+00:00    7221.43
2020-01-01 19:59:59.999999999+00:00    7229.48
2020-01-02 00:59:59.999999999+00:00    7211.02
2020-01-02 05:59:59.999999999+00:00    7138.93
2020-01-02 10:59:59.999999999+00:00    7161.83
2020-01-02 15:59:59.999999999+00:00    7130.98
2020-01-02 20:59:59.999999999+00:00    6948.49
2020-01-03 01:59:59.999999999+00:00    6888.85
Freq: 5H, Name: Close, dtype: float64
```

Each timestamp is now guaranteed to produce a correct resampling operation:

```pycon
>>> h5_close_2d.resample("1d").last()
Close time
2020-01-01 00:00:00+00:00    7229.48
2020-01-02 00:00:00+00:00    6948.49
2020-01-03 00:00:00+00:00    6888.85
Freq: D, Name: Close, dtype: float64
```

!!! note
    Whenever using the close time, don't specify the right bound when resampling with vectorbt methods.
    For instance, instead of using [GenericAccessor.resample_closing](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_closing),
    you're now safe to use [GenericAccessor.resample_opening](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_opening).

## Portfolio

Whenever working with portfolios, we must distinguish between two time frames: the one used 
during simulation and the one used during analysis (or reconstruction). By default, both time frames
are equal. But using a special command, we can execute the trading strategy using a more granular data 
and then downsample the simulated data for analysis. This brings two key advantages:

1. Using a shorter time frame during simulation, we can place a lot more orders more precisely
2. Using a longer time frame during analysis, we can cut down memory consumption and processing time

Let's simulate a simple crossover strategy on `H1` data:

```pycon
>>> fast_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(10))
>>> slow_sma = vbt.talib("SMA").run(h1_close, timeperiod=vbt.Default(20))
>>> entries = fast_sma.real_crossed_above(slow_sma.real)
>>> exits = fast_sma.real_crossed_below(slow_sma.real)

>>> pf = vbt.Portfolio.from_signals(h1_close, entries, exits)
>>> pf.plot()
```

![](/assets/images/tutorials/mtf_analysis_h1_pf.svg)

Computing the returns of a portfolio involves reconstructing many attributes, including 
the cash flow, cash, asset flow, asset value, value, and finally returns. This cascade of 
reconstructions may become a bottleneck if the input data, such as tick data, is too granular.
Luckily, there is a brandnew method [Wrapping.resample](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping.resample),
which allows us to resample vectorbt objects of arbitrary complexity (as long as resampling is 
possible and logically justifiable). Here, we are resampling the portfolio to the start of each month:

```pycon
>>> ms_pf = pf.resample("MS")
>>> ms_pf.plot()
```

![](/assets/images/tutorials/mtf_analysis_ms_pf.svg)

The main artifacts of a simulation are the close price, order records, and additional inputs such as
cash deposits and earnings. Whenever we trigger a resampling job, the close price and those additional
inputs are resampled pretty easily using a bunch of `last` and `sum` operations. 

The order records, on the other hand, are more complex in nature: they are structured NumPy arrays 
(similar to a Pandas DataFrame) that hold order information at each row. The timestamp of each order 
is stored in a separate column of that array, such that we can have multiple orders at the same timestamp. 
This means that we can resample such records simply by re-indexing their timestamp column to the target index,
which is done using [Resampler.map_to_target_index](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler.map_to_target_index).

After resampling the artifacts, a new [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) 
instance is created, and the attributes such as returns are reconstructed on the new data.
This is a perfect example of why vectorbt reconstructs all attributes after the simulation 
and not during the simulation like many conventional backtesters do. 

To prove that we can trust the results:

```pycon
>>> pf.total_return
2.735083772113918

>>> ms_pf.total_return
2.735083772113918
```

Or by comparing the resampled returns of the original portfolio to the returns of the resampled portfolio:

```pycon
>>> (1 + pf.returns).resample("MS").apply(lambda x: x.prod() - 1)
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64

>>> ms_pf.returns
Open time
2020-01-01 00:00:00+00:00    0.150774
2020-02-01 00:00:00+00:00    0.057471
2020-03-01 00:00:00+00:00   -0.005920
2020-04-01 00:00:00+00:00    0.144156
2020-05-01 00:00:00+00:00    0.165367
2020-06-01 00:00:00+00:00   -0.015025
2020-07-01 00:00:00+00:00    0.179079
2020-08-01 00:00:00+00:00    0.084451
2020-09-01 00:00:00+00:00   -0.018819
2020-10-01 00:00:00+00:00    0.064898
2020-11-01 00:00:00+00:00    0.322020
2020-12-01 00:00:00+00:00    0.331068
Freq: MS, Name: Close, dtype: float64
```

!!! hint
    Actually, since returns are reconstructed all the way up from order records and involve 
    so many other attributes, having identical results like this shows that the entire 
    implementation of vectorbt is algorithmically correct :smirk:

BTW If you're wondering how to aggregate those P&L values on the graph, do the following:

```pycon
>>> ms_pf.trades.pnl.reduce_segments(
...     ms_pf.trades.idx_arr, 
...     vbt.nb.sum_reduce_nb
... ).to_pd()  # (1)!
Open time
2020-01-01 00:00:00+00:00     15.077357
2020-02-01 00:00:00+00:00      6.613564
2020-03-01 00:00:00+00:00     -0.113362
2020-04-01 00:00:00+00:00     16.831599
2020-05-01 00:00:00+00:00     22.888280
2020-06-01 00:00:00+00:00     -2.502485
2020-07-01 00:00:00+00:00     26.603047
2020-08-01 00:00:00+00:00     18.804921
2020-09-01 00:00:00+00:00     -6.180621
2020-10-01 00:00:00+00:00     10.133302
2020-11-01 00:00:00+00:00     35.891558
2020-12-01 00:00:00+00:00    129.461217
Freq: MS, Name: Close, dtype: float64
```

1. Extract the P&L values of all exit trade records 
(returned as a [MappedArray](/api/records/mapped_array/#vectorbtpro.records.mapped_array.MappedArray) instance), and 
use [MappedArray.reduce_segments](/api/records/mapped_array/#vectorbtpro.records.mapped_array.MappedArray.reduce_segments)
to aggregate each group of values that map to the same index

## Summary

We should keep in mind that when working with bars, any information stored under a timestamp 
doesn't usually happen exactly at that point in time - it happens somewhere in between this 
timestamp and the next one. This may sound very basic, but this fact changes the resampling logic 
drastically since now we have to be very careful to not catch the look-ahead bias when aligning 
multiple time frames. Gladly, vectorbt implements a range of highly-optimized functions 
that can take this into account and make our lives easier!

[:material-lock: Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/locked-notebooks.md){ .md-button target="blank_" }