---
title: Patterns
---

# :material-lock-open: Patterns and projections - Patterns

Patterns provide a sense of order in what might otherwise appear chaotic. They tend to emerge everywhere 
in nature: shapes like circles in road signs or the rectangles in windows and doors. Just 
like children who gradually learn how to function in a completely unknown environment
by looking for regularities, financial market participants need to learn how to navigate their 
complex, ever-changing environment as well. And just like our little fellows who have parents and teachers 
to assist their learning process (and Google, of course), we have a wonderful tool to assist ours — 
quantitative analysis. 

In this context, patterns are the distinctive formations created by the movements of prices on a 
chart and are the foundation of technical analysis. They help to suggest what prices might do next, 
based on what they have done in the past. For example, we can query all the occurrences of the picture we 
observe today from the past, to analyze how the things developed each time and to make a more nuanced 
trading decision by accounting for various possibilities. Patterns are especially useful in identifying 
points of transition between rising and falling trends, which is the quintessence of successful entry 
and exit timing. But patterns do not give any guarantee to result in the same outcome as before, neither
they last forever: as opposed to the real world with its consistent structure and behavior, the financial 
world is dominated by noise and [false positives](https://en.wikipedia.org/wiki/False_positives_and_false_negatives)
coming from intense interactions of people, systems, and other entities, and so finding something that 
works slightly better than random in a specific market regime is already an achievement - an exciting 
game with probabilities.

Let's create a simple use case where we want to identify the 
[Double Top](https://www.investopedia.com/terms/d/doubletop.asp) pattern.
Let's pull two years of the daily `BTCUSDT` history as our baseline data:

```pycon
>>> data = vbt.BinanceData.fetch(
...     "BTCUSDT", 
...     start="2020-06-01 UTC", 
...     end="2022-06-01 UTC"
... )
>>> data.plot()
```

![](/assets/images/tutorials/patterns/data.svg)

By quickly scanning the picture, we can identify probably the most apparent occurrence of this pattern
between October and December 2021:

```pycon
>>> data_window = data.loc["2021-09-25":"2021-11-25"]
>>> data_window.plot(plot_volume=False)
```

![](/assets/images/tutorials/patterns/data_window.svg)

As human beings, we can easily detect patterns in visual data. But how can it be done programmatically,
where everything revolves around numbers? It would be expensive, unnecessary, and most likely in-effective 
training a [DNN](https://en.wikipedia.org/wiki/Deep_learning) just for this simple task. Following
the wisdom "the simpler the algorithm, the better" for noisy data, we should design an algorithm that 
does the job conventionally, that is, by using loops and basic math.

Since one pattern can only be matched against one feature of data, we will use the 
[typical price](https://en.wikipedia.org/wiki/Typical_price):

```pycon
>>> price_window = data_window.hlc3
>>> price_window.vbt.plot()
```

![](/assets/images/tutorials/patterns/hlc3.svg)

Let's design the pattern first. Price patterns are identified using a series of lines and/or curves.
The "Double Top" pattern, for instance, can be represented by the following array:

```pycon
>>> pattern = np.array([1, 2, 3, 2, 3, 2])
>>> pd.Series(pattern).vbt.plot()
```

![](/assets/images/tutorials/patterns/double_top_pattern.svg)

It's important to note that the length and the absolute values of the pattern definition are of no importance;
if the pattern runs from 1 to 3 or 1 to 60 will not make any difference because the pattern will be
stretched horizontally and vertically to fit the respective price. What affects the computation
though is how individual points are located in relation to each other. For example, the first point `2`
comes exactly in the middle between the points `1` and `3`, meaning if the asset jumped in price to match 
the second point, it would ideally make the same jump to reach the first peak. Furthermore, since
some numbers such as `2` and `3` repeat, we don't expect the price at those points to deviate much,
which is useful for defining support/resistance lines. 

Another important rule is concerned around the horizontal pattern structure: irrespective of the value 
at any point, the location (timing) of the point is also relative to the location of the surrounding points. 
For example, if the first point was matched on `2020-01-01` and the second point on `2020-01-03`, the
third point is expected to match on `2020-01-06`. This also means that if any part of the pattern 
requires more time to develop and thus changes the horizontal structure, matching would become less likely.

## Interpolation

After we have defined our pattern, we need to bring it and the price to the same length.
In image processing, increasing the size of an image requires reconstruction of the image by 
[interpolating](https://en.wikipedia.org/wiki/Interpolation) new pixels. Reducing the size of an 
image requires [downsampling](https://en.wikipedia.org/wiki/Downsampling_(signal_processing)) of the 
existing pixels. In pattern processing, the approach is similar; the only difference is that we're 
working on one-dimensional arrays instead of two-dimensional. We also prefer interpolation 
(stretching) over downsampling (shrinking) to avoid information loss. This means that if the price
is smaller than the pattern, it should be stretched to match the length of the pattern
rather than compressing the pattern to match the length of the price.

There are four main interpolation modes in pattern processing: linear, nearest neighbor, discrete, and mixed.
All of them are implemented using the Numba-compiled function 
[interp_resize_1d_nb](/api/generic/nb/patterns/#vectorbtpro.generic.nb.patterns.interp_resize_1d_nb),
which takes an array, a target size, and an interpolation mode of the type 
[InterpMode](/api/generic/enums/#vectorbtpro.generic.enums.InterpMode). The implementation
is highly efficient: it goes through the array only once and doesn't require creation of additional 
arrays apart from the final array.

### Linear

The linear interpolation algorithm involves estimating a new value by connecting two adjacent known 
values with a straight line (see [here](https://scientific-python.readthedocs.io/en/latest/notebooks_rst/1_Interpolation/1D_interpolation.html#linear-interpolation)
for an illustration). 

Let's stretch our pattern array to 10 points:

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, 10, vbt.enums.InterpMode.Linear
... )
>>> resized_pattern
array([1.        , 1.55555556, 2.11111111, 2.66666667, 2.77777778,
       2.22222222, 2.33333333, 2.88888889, 2.55555556, 2.        ])
```

This mode works best if the target size is close to `2 * n - 1`, where `n` is the original size of the array.
In such a case, the characteristics of the resized array will closely match that of the original array.
Otherwise, the relation of points to each other will be violated, unless the target size is sufficiently 
bigger than the original size. This is best demonstrated below, where we resize the array to the length of 
7, 11, and 30 points respectively:

```pycon
>>> def plot_linear(n):
...     resized_pattern = vbt.nb.interp_resize_1d_nb(
...         pattern, n, vbt.enums.InterpMode.Linear
...     )
...     return pd.Series(resized_pattern).vbt.plot()
```

=== "7 points"

    ```pycon
    >>> plot_linear(7)
    ```

    ![](/assets/images/tutorials/patterns/linear_7.svg)

=== "11 points"

    ```pycon
    >>> plot_linear(11)
    ```

    ![](/assets/images/tutorials/patterns/linear_11.svg)

=== "30 points"

    ```pycon
    >>> plot_linear(30)
    ```

    ![](/assets/images/tutorials/patterns/linear_30.svg)

Why is the first graph so "ugly"? Because the algorithm needs to keep the same distance between each
pair of points in the resized array. To stretch a 6-point array to 7 points, the algorithm 
first needs to split the graph of the 6-point array into 7 parts, and then to select the value
that's located at the mid-point of each part, like this:

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, 7, vbt.enums.InterpMode.Linear
... )
>>> ratio = (len(pattern) - 1) / (len(resized_pattern) - 1)
>>> new_points = np.arange(len(resized_pattern)) * ratio
>>> fig = pd.Series(pattern).vbt.plot()
>>> pd.Series(resized_pattern, index=new_points).vbt.scatterplot(fig=fig)
```

![](/assets/images/tutorials/patterns/linear_7_scatter.svg)

As we can see, linear interpolation is all about selecting a specific number of equidistant values from 
the original array, and the greater is the number of points, the better is the result. The main
issue is that whenever the target size is suboptimal, the scale of the resized array will change.
In the example above, the pattern will be distorted and the original link between the points 
with the value `2` will be gone. But there are other interpolation algorithms that do better here.

### Nearest

The nearest neighbor algorithm selects the value of the nearest point and does not consider 
the values of other points at all (see [here](https://scientific-python.readthedocs.io/en/latest/notebooks_rst/1_Interpolation/1D_interpolation.html#nearest-aka-piecewise-interpolation)
for an illustration). This way, the resized array will consist exclusively of the values
in the original array and there won't be any intermediate values as floating numbers present:

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, 10, vbt.enums.InterpMode.Nearest
... )
>>> resized_pattern
array([1., 2., 2., 3., 3., 2., 2., 3., 3., 2.])
```

!!! info
    The resized array will always be floating for consistency reasons.

And here are resized arrays for different target sizes:

```pycon
>>> def plot_nearest(n):
...     resized_pattern = vbt.nb.interp_resize_1d_nb(
...         pattern, n, vbt.enums.InterpMode.Nearest
...     )
...     return pd.Series(resized_pattern).vbt.plot()
```

=== "7 points"

    ```pycon
    >>> plot_nearest(7)
    ```

    ![](/assets/images/tutorials/patterns/nearest_7.svg)

=== "11 points"

    ```pycon
    >>> plot_nearest(11)
    ```

    ![](/assets/images/tutorials/patterns/nearest_11.svg)

=== "30 points"

    ```pycon
    >>> plot_nearest(30)
    ```

    ![](/assets/images/tutorials/patterns/nearest_30.svg)

As we can see, each graph above is basically a step curve consisting of horizontal and vertical lines.
Because of this, it may be challenging to apply since we expect the price to change gradually rather 
to jump around sharply, thus this interpolation mode should be used only when the original array is 
granular enough to smoothen the transitions between local extrema.

!!! hint
    The `2 * n - 1` rule doesn't hold for this mode.

### Discrete

The discrete interpolation algorithm selects the value depending on whether it's closer to the 
target position than other values, and if not, sets it to NaN. This mode guarantees to produce an 
array with values taken from the original array only once, but may still change their temporal distribution.
This makes most sense in scenarios where the transition between each pair of points is of no interest.

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, 10, vbt.enums.InterpMode.Discrete
... )
>>> resized_pattern
array([ 1., nan,  2., nan,  3.,  2., nan,  3., nan,  2.])
```

Here's a comparison of differently-resized arrays:

```pycon
>>> def plot_discrete(n):
...     resized_pattern = vbt.nb.interp_resize_1d_nb(
...         pattern, n, vbt.enums.InterpMode.Discrete
...     )
...     return pd.Series(resized_pattern).vbt.plot(
...         trace_kwargs=dict(
...             line=dict(dash="dot"), 
...             connectgaps=True
...         )
...     )
```

=== "7 points"

    ```pycon
    >>> plot_discrete(7)
    ```

    ![](/assets/images/tutorials/patterns/discrete_7.svg)

=== "11 points"

    ```pycon
    >>> plot_discrete(11)
    ```

    ![](/assets/images/tutorials/patterns/discrete_11.svg)

=== "30 points"

    ```pycon
    >>> plot_discrete(30)
    ```

    ![](/assets/images/tutorials/patterns/discrete_30.svg)


Each of the graphs contains exactly 6 points that have been mapped to a new interval by trying
to keep the distance between them as equal as possible. Similarly to the linear interpolation, this mode 
also yields the best results only if the target size is `2 * n - 1`, while other sizes distort the temporal 
distribution of the points. In contrast to the linear interpolation though, this mode respects the 
absolute values of the original array such that the point `2` is always guaranteed to be on the 
midway between the points `1` and `3`.

### Mixed

The mixed interpolation algorithm is a mix of the linear and discrete interpolation algorithms.
First, it calls the discrete interpolator, and if the value is NaN, it calls the linear interpolator.
This way, we are guaranteed to include each value from the original array at least once to keep
the original scaling, and at the same time connect them by linearly interpolating the intermediate values - 
the best of both worlds.

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, 10, vbt.enums.InterpMode.Mixed
... )
>>> resized_pattern
array([1.        , 1.55555556, 2.        , 2.66666667, 3.        ,
       2.        , 2.33333333, 3.        , 2.55555556, 2.        ])
```

Let's demonstrate how the mixed approach "fixes" the scale problem of the linear approach:

```pycon
>>> def plot_mixed(n):
...     lin_resized_pattern = vbt.nb.interp_resize_1d_nb(
...         pattern, n, vbt.enums.InterpMode.Linear
...     )
...     mix_resized_pattern = vbt.nb.interp_resize_1d_nb(
...         pattern, n, vbt.enums.InterpMode.Mixed
...     )
...     fig = pd.Series(lin_resized_pattern, name="Linear").vbt.plot()
...     return pd.Series(mix_resized_pattern, name="Mixed").vbt.plot(fig=fig)
```

=== "7 points"

    ```pycon
    >>> plot_mixed(7)
    ```

    ![](/assets/images/tutorials/patterns/mixed_7.svg)

=== "11 points"

    ```pycon
    >>> plot_mixed(11)
    ```

    ![](/assets/images/tutorials/patterns/mixed_11.svg)

=== "30 points"
    
    ```pycon
    >>> plot_mixed(30)
    ```

    ![](/assets/images/tutorials/patterns/mixed_30.svg)

!!! info
    As you probably noticed, the last pattern line is not entirely straight on the graph.
    This is because by using the mixed interpolation we're interpolating some points linearly
    and some discretely to retain the original scale. Producing a clean line would require us
    to go through the data more than once, thus we chose performance over visual aesthetics.

We have restored the original connection between various points, hence this algorithm should be (and is) 
the default choice when it comes to interpolation without gaps.

```pycon
>>> resized_pattern = vbt.nb.interp_resize_1d_nb(
...     pattern, len(price_window), vbt.enums.InterpMode.Mixed
... )
>>> resized_pattern.shape
(62,)
```

## Rescaling

After we brought the pattern and the price to the same length, we need to bring
them to the same scale as well in order to make them comparable. For this, we need to compute 
the minimum and maximum of the pattern and price, and rescale the pattern to match the scale 
of the price. We can use the Numba-compiled function [rescale_nb](/api/utils/array_/#vectorbtpro.utils.array_.rescale_nb), 
which takes an array, the scale of the array, and the target scale:

```pycon
>>> pattern_scale = (resized_pattern.min(), resized_pattern.max())
>>> price_window_scale = (price_window.min(), price_window.max())
>>> rescaled_pattern = vbt.utils.array_.rescale_nb(
...     resized_pattern, pattern_scale, price_window_scale
... )
>>> rescaled_pattern = pd.Series(rescaled_pattern, index=price_window.index)
```

We can now finally overlay the pattern over the price:

```pycon
>>> fig = price_window.vbt.plot()
>>> rescaled_pattern.vbt.plot(
...     trace_kwargs=dict(
...         fill="tonexty", 
...         fillcolor="rgba(255, 0, 0, 0.25)"
...     ), 
...     fig=fig
... )
```

![](/assets/images/tutorials/patterns/rescaled_pattern.svg)

### Rebasing

Another way to bring both arrays to the same scale is by using rebasing, which makes
the first value of both arrays equal, and all other values are rescaled using their relative distance to 
the starting point. This is useful in the case where our pattern should also enforce a certain percentage 
change from the starting point. For example, let's enforce the relative distance between the peak and the 
starting point of 60%:

```pycon
>>> pct_pattern = np.array([1, 1.3, 1.6, 1.3, 1.6, 1.3])
>>> resized_pct_pattern = vbt.nb.interp_resize_1d_nb(
...     pct_pattern, len(price_window), vbt.enums.InterpMode.Mixed
... )
>>> rebased_pattern = resized_pct_pattern / resized_pct_pattern[0]
>>> rebased_pattern *= price_window.values[0]
>>> rebased_pattern = pd.Series(rebased_pattern, index=price_window.index)
>>> fig = price_window.vbt.plot()
>>> rebased_pattern.vbt.plot(
...     trace_kwargs=dict(
...         fill="tonexty", 
...         fillcolor="rgba(255, 0, 0, 0.25)"
...     ), 
...     fig=fig
... )
```

![](/assets/images/tutorials/patterns/rebased_pattern.svg)

## Similarity

Our arrays are now perfectly comparable, so how do we calculate their similarity? The algorithm
is rather easy: compute the absolute, element-wise distances between the values of both arrays,
and add them up (a.k.a. [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) or MAE). 
At the same time, compute the maximum possible absolute, element-wise distances,
and add them up. The maximum distance is calculated relative to the global minimum
and maximum value. Finally, divide both totals and subtract from 1 to get the similarity score
that ranges between 0 and 1:

```pycon
>>> abs_distances = np.abs(rescaled_pattern - price_window.values)
>>> mae = abs_distances.sum()
>>> max_abs_distances = np.column_stack((
...     (price_window.max() - rescaled_pattern), 
...     (rescaled_pattern - price_window.min())
... )).max(axis=1)
>>> max_mae = max_abs_distances.sum()
>>> similarity = 1 - mae / max_mae
>>> similarity
0.8726845123416802
```

To penalize large distances and make the pattern detection more "strict",
we can switch the distance measure to the root of sum of squared distances (a.k.a. 
[Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) or RMSE):

```pycon
>>> quad_distances = (rescaled_pattern - price_window.values) ** 2
>>> rmse = np.sqrt(quad_distances.sum())
>>> max_quad_distances = np.column_stack((
...     (price_window.max() - rescaled_pattern), 
...     (rescaled_pattern - price_window.min())
... )).max(axis=1) ** 2
>>> max_rmse = np.sqrt(max_quad_distances.sum())
>>> similarity = 1 - rmse / max_rmse
>>> similarity
0.8484851233108504
```

As a further adaptation, we could have also removed the root from the equation above and 
calculated just the sum of squared distances (a.k.a. 
[Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) or MSE):

```pycon
>>> quad_distances = (rescaled_pattern - price_window.values) ** 2
>>> mse = quad_distances.sum()
>>> max_quad_distances = np.column_stack((
...     (price_window.max() - rescaled_pattern), 
...     (rescaled_pattern - price_window.min())
... )).max(axis=1) ** 2
>>> max_mse = max_quad_distances.sum()
>>> similarity = 1 - mse / max_mse
>>> similarity
0.9770432421418718
```

!!! note
    Since the maximum distance is now the power of 2 of the absolute maximum distance, the similarity 
    will often cross the mark of 90% and above, thus do not forget to adapt your thresholds as well.

If you're frightened of writing the code above each time you need to measure the similarity between
two arrays, don't! As with everything, there is the convenient Numba-compiled function 
[pattern_similarity_nb](/api/generic/nb/patterns/#vectorbtpro.generic.nb.patterns.pattern_similarity_nb), 
which combines all the steps above to produce a single number. This function accepts many options
for interpolation, rescaling, and distance measurement, and runs in [O(n)](https://en.wikipedia.org/wiki/Time_complexity) 
time without creating any new arrays. Due to its exceptional efficiency and compilation with Numba, 
we can run the function millions of times in a fraction of a second :fire: The only difference to our 
approach above is that it rescales the price array to the pattern scale, not the other way around 
(which we used for plotting reasons).

Let's explore the power of this function by replicating our pipeline above: 

```pycon
>>> vbt.nb.pattern_similarity_nb(price_window.values, pattern)
0.8726845123416802
```

The same score as we produced manually :eyes:

Let's calculate the similarity score for `pct_pattern` with rebasing:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pct_pattern, 
...     rescale_mode=vbt.enums.RescaleMode.Rebase  # (1)!
... )
0.8647140967291362
```

1. See [RescaleMode](/api/generic/enums/#vectorbtpro.generic.enums.RescaleMode)

Let's get the similarity score from interpolating the pattern using the nearest-neighbor
interpolation, rebasing, and RMSE:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pct_pattern, 
...     interp_mode=vbt.enums.InterpMode.Nearest,  # (1)!
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     distance_measure=vbt.enums.DistanceMeasure.RMSE  # (2)!
... )
0.76151009787845
```

1. See [RescaleMode](/api/generic/enums/#vectorbtpro.generic.enums.InterpMode)
2. See [DistanceMeasure](/api/generic/enums/#vectorbtpro.generic.enums.DistanceMeasure)

Since often we're not only interested in getting the similarity measure but also in being able to visualize and 
debug the pattern, we can call the accessor method [GenericSRAccessor.plot_pattern](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericSRAccessor.plot_pattern),
which reconstructs the similarity calculation and displays various artifacts visually. That is,
if we want it to produce an accurate plot, we need to provide the same arguments as we provide 
to the similarity calculation function. Our last example produced a similarity of 75%, let's
visualize the fit:

```pycon
>>> price_window.vbt.plot_pattern(
...     pct_pattern, 
...     interp_mode="nearest",  # (1)!
...     rescale_mode="rebase",
...     fill_distance=True
... )
```

1. We can provide each option as a string here

![](/assets/images/tutorials/patterns/plot_pattern.svg)

We see that the biggest discrepancy comes at the valley in the middle: the interpolated pattern expects
the price to dip deeper than it actually does. Let's add 15% to that point to increase the similarity:

```pycon
>>> adj_pct_pattern = np.array([1, 1.3, 1.6, 1.45, 1.6, 1.3])
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     adj_pct_pattern, 
...     interp_mode=vbt.enums.InterpMode.Nearest,
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     distance_measure=vbt.enums.DistanceMeasure.RMSE
... )
0.8086016654243109
```

And here's how the discrete interpolation applied to the new pattern looks like:

```pycon
>>> price_window.vbt.plot_pattern(
...     adj_pct_pattern, 
...     interp_mode="discrete",
...     rescale_mode="rebase",
... )
```

![](/assets/images/tutorials/patterns/plot_pattern_discrete.svg)

The pattern trace has become a scatter plot rather than a line plot because the similarity is
calculated based solely on those points whereas the greyed out points are ignored. If we 
calculated the similarity score once again, we would see a number higher than previously
because the pattern at those points matches the price pretty accurately:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     adj_pct_pattern, 
...     interp_mode=vbt.enums.InterpMode.Discrete,
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     distance_measure=vbt.enums.DistanceMeasure.RMSE
... )
0.8719692914480557
```

### Relative

Since the price is not static, and it may change significantly during the period of comparison,
we should be interested in calculating the relative as opposed to the absolute distance (error). For example,
if the first price point is `10` and the last price point is `1000`, the distance to the latter would 
have a much greater impact on the similarity score than the distance to the former. Let's re-calculate 
the score manually and automatically using relative distances:

```pycon
>>> abs_pct_distances = abs_distances / rescaled_pattern
>>> pct_mae = abs_pct_distances.sum()
>>> max_abs_pct_distances = max_abs_distances / rescaled_pattern
>>> max_pct_mae = max_abs_pct_distances.sum()
>>> similarity = 1 - pct_mae / max_pct_mae
>>> similarity
0.8732697724295595

>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pct_pattern, 
...     error_type=vbt.enums.ErrorType.Relative  # (1)!
... )
0.8732697724295594
```

1. See [ErrorType](/api/generic/enums/#vectorbtpro.generic.enums.ErrorType)

The difference is not that big in our scenario, but here's what happens when the price 
moves sharply:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     np.array([10, 30, 100]),
...     np.array([1, 2, 3]),
...     error_type=vbt.enums.ErrorType.Absolute
... )
0.8888888888888888

>>> vbt.nb.pattern_similarity_nb(
...     np.array([10, 30, 100]),
...     np.array([1, 2, 3]),
...     error_type=vbt.enums.ErrorType.Relative
... )
0.9575911789652247
```

In both examples, the pattern has been rescaled to `[1, 50, 100]` using the min-max rescaler (default).
In first example though, the normalized error is `abs(30 - 50) / (100 - 30) = 0.29`, while in the second
example the normalized error is `(abs(30 - 50) / 50) / ((100 - 30) / 30) = 0.17`, which also takes into 
account the volatility of the price.

### Inverse

We can also invert the pattern internally:

```pycon
>>> vbt.nb.pattern_similarity_nb(price_window.values, pattern, invert=True)
0.32064009029620244

>>> price_window.vbt.plot_pattern(pattern, invert=True)
```

![](/assets/images/tutorials/patterns/inverted_pattern.svg)

!!! note
    This isn't the same as simply inverting the score.

To produce the inverted pattern manually:

```pycon
>>> pattern.max() + pattern.min() - pattern
array([3, 2, 1, 2, 1, 2])
```

### Max error

Sometimes we may want to define patterns as "corridors" within which the price should move.
If any of the corridor points were violated, we can either set the distance at that point
to the maximum distance (`max_error_strict=False`), or set the entire similarity to NaN 
(`max_error_strict=True`). Such a corridor is referred to as "maximum error". This error
can be provided thorugh the array-like argument `max_error`, which should be defined in the same 
way as the pattern; that is, it mostly needs to have the same length and scale as the pattern.

For example, if we chose the min-max rescaling and the pattern was defined from `1` to `6`,
a maximum error of `0.5` would be `0.5 / (6 - 1) = 0.1`, that is, 10% of the pattern's scale.
Let's query the similarity of our original pattern without and with a corridor of `0.5`:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern,
... )
0.8726845123416802

>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern, 
...     max_error=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
... )
0.8611332262389184
```

Since `max_error` is a flexible argument, we can also provide it as a zero-dimensional or 
one-dimensional array with one value, which will be valid for each point in the pattern:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern, 
...     max_error=np.array([0.5]),  # (1)!
... )
0.8611332262389184
```

1. The argument cannot be provided as a constant (`0.5`), it must always be an array

The similarity score has decreased, which means that some corridor points were violated.
Let's visualize the entire thing to see where exactly:

```pycon
>>> price_window.vbt.plot_pattern(
...     pattern, 
...     max_error=0.5  # (1)!
... )
```

1. We can provide constants here

![](/assets/images/tutorials/patterns/max_error.svg)

We can see two points that were violated, thus their distance to the price was set to the maximum
possible distance, which brought the similarity down. If we enabled the strict mode though,
the similarity would have become NaN to notify the user that it didn't pass the test:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern, 
...     max_error=np.array([0.5]),
...     max_error_strict=True
... )
nan
```

If we're interested in relative distances using `ErrorType.Relative`, the maximum error should
be provided as a percentage change:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern, 
...     max_error=np.array([0.1]),  # (1)!
...     error_type=vbt.enums.ErrorType.Relative
... )
0.8548520433078988

>>> price_window.vbt.plot_pattern(
...     pattern, 
...     max_error=0.1,
...     error_type="relative"
... )
```

1. Corridor of 10% at any point

![](/assets/images/tutorials/patterns/max_error_relative.svg)

The same goes for the rescaling using rebasing mode, where irrespective of the error type each error 
must be given as a percentage change. For example, if the current pattern value has been mapped the 
price of `12000` and the maximum error is `0.1`, the corridor will encompass all the values from 
`12000 * 0.9 = 10800` to `12000 * 1.1 = 13200`. Let's permit deviations for the adjusted percentage 
pattern of no more than 20% at the first level, 10% at the second level, and 5% at the third level:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     adj_pct_pattern, 
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     max_error=np.array([0.2, 0.1, 0.05, 0.1, 0.05, 0.1]),
...     max_error_strict=True
... )
nan

>>> price_window.vbt.plot_pattern(
...     adj_pct_pattern, 
...     rescale_mode="rebase",
...     max_error=np.array([0.2, 0.1, 0.05, 0.1, 0.05, 0.1])
... )
```

![](/assets/images/tutorials/patterns/max_error_rebase.svg)

As we can see, some points go outside the corridor. If we added additional 5% to all points,
the pattern would pass the test easily:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     adj_pct_pattern, 
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     max_error=np.array([0.2, 0.1, 0.05, 0.1, 0.05, 0.1]) + 0.05,
...     max_error_strict=True
... )
0.8789689066239321
```

#### Interpolation

But is there a way to provide the maximum error discretely? That is, can we force the function to adhere
to the corridor at certain points rather than at all points gradually? By default, the 
maximum error gets interpolated the same way as the pattern (linearly in our case).
To make the maximum error array interpolate differently, provide a different mode as `max_error_interp_mode`.
For example, let's only force the peak points to be within the corridor of 10%. For this,
we need to use the discrete interpolation mode and set all the intermediate points to NaN:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     adj_pct_pattern, 
...     rescale_mode=vbt.enums.RescaleMode.Rebase,
...     max_error=np.array([np.nan, np.nan, 0.1, np.nan, 0.1, np.nan]),
...     max_error_interp_mode=vbt.enums.InterpMode.Discrete,
...     max_error_strict=True
... )
0.8789689066239321

>>> price_window.vbt.plot_pattern(
...     adj_pct_pattern, 
...     rescale_mode="rebase",
...     max_error=np.array([np.nan, np.nan, 0.1, np.nan, 0.1, np.nan]),
...     max_error_interp_mode="discrete"
... )
```

![](/assets/images/tutorials/patterns/max_error_discrete.svg)

Even though the scatter points of the maximum error are connected by a greyed-out line, there is no 
requirement for the price to be between those lines, only between each pair of purple points.

#### Max distance

Final question: is there a way to tweak the maximum distance? Yes! We can use the maximum 
error as the maximum distance by enabling `max_error_as_maxdist`. This has the following implications:
the smaller is the maximum distance at any point, the heavier the volatility of the price at that point 
affects the similarity. Let's compare our original pattern without and with the maximum distance cap
of `0.5` (10% of the scale):

```pycon
>>> vbt.nb.pattern_similarity_nb(price_window.values, pattern)
0.8726845123416802

>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, 
...     pattern, 
...     max_error=np.array([0.5]),
...     max_error_as_maxdist=True
... )
0.6193594883412921
```

This way, we introduced a penalty for a heightened volatility of the price.

!!! note
    You can also set a different maximum distance at different points in the pattern.
    Just note that points with a larger maximum distance will have more weight in the similarity 
    calculation than points with a smaller maximum distance. Consider a scenario where there are two points
    with the maximum distance of `100` and `1` respectively. Even if we had a perfect match at the second 
    point, the similarity would be largely based on the distance at the first point. 

### Further filters

When matching a huge amount of price windows against a pattern, we may want to skip some windows
due to a volatility that is either too low or too high. This is possible by setting the arguments 
`min_pct_change` and `max_pct_change` respectively:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, pattern, max_pct_change=0.3
... )
nan
```

A nice side effect of this is an increased performance: if the test fails, the price window will be 
traversed only once to get the minimum and maximum value.

We can also filter out similarity scores that are below some predefined threshold.
For example, let's set the minimum similarity to 90%:

```pycon
>>> vbt.nb.pattern_similarity_nb(
...     price_window.values, pattern, min_similarity=0.9
... )
nan
```

!!! hint
    Don't be afraid of NaNs, they simply mean *"didn't pass some tests, should be ignored during analysis"*.

Setting a similarity threshold has also a performance benefit: if at some point the algorithm notices 
that the threshold cannot be reached anymore, even if the remaining points had matched perfectly,
it will abort and set the final score to NaN. Depending on the threshold, this makes the computation 
30% faster on average.

## Rolling similarity

We've learned the theory behind pattern recognition, now it's time to get our hands dirty.
To search a price space for a pattern, we need to roll a window over that space. This can be accomplished 
using the accessor method [GenericAccessor.rolling_pattern_similarity](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.rolling_pattern_similarity),
which takes the same arguments as we used before, but also the length of the window to roll.
If the length is `None`, it will be set to the length of the pattern array.

Let's roll a window of 30 data points over the entire typical price, and match against a pattern
that has a discrete soft corridor of 5%:

```pycon
>>> price = data.hlc3

>>> similarity = price.vbt.rolling_pattern_similarity(
...     pattern, 
...     window=30,
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete"
... )
>>> similarity.describe()
count    701.000000
mean       0.499321
std        0.144088
min        0.148387
25%        0.394584
50%        0.502231
75%        0.607962
max        0.838393
dtype: float64
```

We see that among 701 comparisons roughly a half has produced a score below 50%. 
The highest score sits at around 84%. Let's visualize the best match:

```pycon
>>> end_row = similarity.argmax() + 1  # (1)!
>>> start_row = end_row - 30
>>> fig = data.iloc[start_row:end_row].plot(plot_volume=False)
>>> price.iloc[start_row:end_row].vbt.plot_pattern(
...     pattern, 
...     error_type="relative",  # (1)!
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     plot_obj=False, 
...     fig=fig
... )
```

1. Get the start (including) and end (excluding) row of the window with the highest similarity
2. Don't forget to use the same parameters that were used for calculating the similarity

![](/assets/images/tutorials/patterns/max_rolling_similarity.svg)

Pretty accurate, right? And this window matches even better than the window we investigated previously.
But what about the lowest similarity, is it the same as inverting the pattern? No!

```pycon
>>> end_row = similarity.argmin() + 1
>>> start_row = end_row - 30
>>> fig = data.iloc[start_row:end_row].plot(plot_volume=False)
>>> price.iloc[start_row:end_row].vbt.plot_pattern(
...     pattern, 
...     invert=True,
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     plot_obj=False, 
...     fig=fig
... )
```

![](/assets/images/tutorials/patterns/min_rolling_similarity.svg)

Inverting the score will invalidate all the requirements that we put initially, thus you should
always start a new pattern search with the `invert` flag enabled:

```pycon
>>> inv_similarity = price.vbt.rolling_pattern_similarity(
...     pattern, 
...     window=30,
...     invert=True,  # (1)!
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete"
... )
>>> end_row = inv_similarity.argmax() + 1  # (2)!
>>> start_row = end_row - 30
>>> fig = data.iloc[start_row:end_row].plot(plot_volume=False)
>>> price.iloc[start_row:end_row].vbt.plot_pattern(
...     pattern, 
...     invert=True,
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     plot_obj=False, 
...     fig=fig
... )
```

1. Here
2. We're looking for the highest similarity

![](/assets/images/tutorials/patterns/inv_rolling_similarity.svg)

The best match isn't exactly a good fit, but still much better than the previous one.

### Indicator

Once we've settled the optimal pattern parameters through exploration and debugging,
we should start concerning ourselves with integrating the pattern detection component into 
our backtesting stack. This can be done through running the process inside an indicator.
Since indicators must return output arrays of the same shape as their input arrays, we can safely 
use the rolling pattern similarity as output. For this, we can use the
[PATSIM](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM) indicator, which takes
a price array as the only input, and all the arguments related to calculating the pattern 
similarity as parameters, including the pattern array itself! Another advantage of this indicator
is the ability to automatically convert arguments provided as a string (such as `interp_mode`)
into a Numba-compatible format. Finally, indicators are great for testing many windows
as the accuracy of pattern detection heavily depends on the choice of window length.

Let's test multiple window combinations along with the same setup as above:

```pycon
>>> patsim = vbt.PATSIM.run(
...     price, 
...     vbt.Default(pattern),  # (1)!
...     error_type=vbt.Default("relative"),
...     max_error=vbt.Default(0.05),
...     max_error_interp_mode=vbt.Default("discrete"),
...     window=[30, 45, 60, 75, 90]
... )
```

1. Wrap with [Default](/api/base/reshaping/#vectorbtpro.base.reshaping.Default)
to hide the parameter from columns

We can plot the similarity development using the method 
[PATSIM.plot](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM.plot).
As any other plotting method, it allows only one column to be plotted, thus we need
to specify the column name beforehand using the argument `column`:

```pycon
>>> patsim.wrapper.columns  # (1)!
Int64Index([30, 45, 60, 75, 90], dtype='int64', name='patsim_window')

>>> patsim.plot(column=60)
```

1. Take a look at the column names this indicator has generated. If there are multiple column
levels, you need to provide the column name as a tuple.

![](/assets/images/tutorials/patterns/patsim.svg)

The generated similarity series is as fascinating as the price series itself, and
can be used in all sorts of technical analysis on its own :triangular_ruler:

But probably an even more informative plot can be produced by 
[PATSIM.overlay_with_heatmap](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM.overlay_with_heatmap),
which overlays a price line with a similarity heatmap:

```pycon
>>> patsim.overlay_with_heatmap(column=60)
```

![](/assets/images/tutorials/patterns/patsim_heatmap.svg)

!!! info
    Bright vertical lines on the graph are located at the very end of their windows, that is,
    where the pattern is marked as completed. Hence, the similarity score is safe to use in backtesting.

So, how do we use this indicator in signal generation? We can compare the resulting similarity score
to a threshold to derive signals. For our example above, let's set a threshold of 80%
to build the exit signals:

```pycon
>>> exits = patsim.sim >= 0.9
>>> exits.sum()
patsim_window
30     6
45     8
60    14
75     0
90     5
dtype: int64
```

If we wanted to test multiple thresholds, we could have also used the parameter `min_similarity`,
which would set all scores falling below it to NaN, but also make pattern recognition faster on average. 
Deriving the signals would be as simple as checking whether each element is NaN. We'll additionally
test the inverted pattern:

```pycon
>>> patsim = vbt.PATSIM.run(
...     price, 
...     vbt.Default(pattern),
...     error_type=vbt.Default("relative"),
...     max_error=vbt.Default(0.05),
...     max_error_interp_mode=vbt.Default("discrete"),
...     window=[30, 45, 60, 75, 90],
...     invert=[False, True],
...     min_similarity=[0.7, 0.8],
...     param_product=True  # (1)!
... )
>>> exits = ~patsim.sim.isnull()  # (2)!
>>> exits.sum()
patsim_window  patsim_invert  patsim_min_similarity
30             False          0.7                      68
                              0.8                       6
               True           0.7                      64
                              0.8                       2
...            ...            ...                     ...
90             False          0.7                      61
                              0.8                       5
               True           0.7                      70
                              0.8                       8
dtype: int64
```

1. Build the Cartesian product of all parameters with more than one element. 
Beware that the number of parameter combinations can grow rapidly: testing three parameters with 
10 elements each would result in 1000 combinations!
2. Any element that is not NaN is a potential signal

What if we're not interested in having the window as a backtestable parameter, but rather we want to 
create a signal as soon as any of the windows at that row crossed the similarity threshold? This way, 
we would be able to react immediately once a pattern of any length was detected. This is easily achievable
using Pandas:

```pycon
>>> groupby = [  # (1)!
...     name for name in patsim.wrapper.columns.names 
...     if name != "patsim_window"
... ]
>>> max_sim = patsim.sim.groupby(groupby, axis=1).max()  # (2)!
>>> entries = ~max_sim.xs(True, level="patsim_invert", axis=1).isnull()  # (3)!
>>> exits = ~max_sim.xs(False, level="patsim_invert", axis=1).isnull()  # (4)!
```

1. Group by each column level except `patsim_window`
2. Select the highest score among all windows
3. Select non-NaN scores among columns where `patsim_invert` is `True` to build entries
4. Select non-NaN scores among columns where `patsim_invert` is `False` to build exits

Let's plot the entry and exit signals corresponding to the threshold of 80%:

```pycon
>>> fig = data.plot(ohlc_trace_kwargs=dict(opacity=0.5))
>>> entries[0.8].vbt.signals.plot_as_entries(price, fig=fig)
>>> exits[0.8].vbt.signals.plot_as_exits(price, fig=fig)
```

![](/assets/images/tutorials/patterns/patsim_signals.svg)

Apart from a few failed regular and inverted double top patterns, our indicator does great. 
By further tweaking the pattern similarity parameters and choosing a somewhat more strict pattern 
configuration, we could easily filter out most failed patterns.

## Search

Searching for patterns of a variable length using indicators with a parametrizable window is 
expensive: each window would require allocation of an array of *at least* the same shape as the entire 
price array. We need a more compressed representation of a pattern search result.
Gladly, the vectorbt's native support for record arrays makes exactly this possible! 

The procedural logic is implemented by the Numba-compiled function 
[find_pattern_1d_nb](/api/generic/nb/records/#vectorbtpro.generic.nb.records.find_pattern_1d_nb)
and its two-dimensional version [find_pattern_nb](/api/generic/nb/records/#vectorbtpro.generic.nb.records.find_pattern_nb).
The idea is the following: iterate over the rows of a price array, and at each row, 
iterate over a range of windows headed forward/backward. For each window, run the
[pattern_similarity_nb](/api/generic/nb/patterns/#vectorbtpro.generic.nb.patterns.pattern_similarity_nb)
function to get the similarity score. If the score passes all requirements and thus is not NaN,
create a NumPy record of the type [pattern_range_dt](/api/generic/enums/#vectorbtpro.generic.enums.pattern_range_dt),
which stores the start index (including), the end index (excluding or including if it's the last index),
and the similarity score. This record gets appended to a record array and returned to the user.
The function is capable of selecting rows and windows randomly given a certain probability to decrease
the number of candidates, and handling overlapping pattern ranges. Also, it's incredibly 
time and memory efficient :zap:

Let's search for the pattern we defined above by rolling a window of the size from 30 to 90,
and requiring the similarity score to be at least 85%:

```pycon
>>> pattern_range_records = vbt.nb.find_pattern_1d_nb(
...     price.values,  # (1)!
...     pattern,
...     window=30,  # (2)!
...     max_window=90,
...     error_type=vbt.enums.ErrorType.Relative,
...     max_error=np.array([0.05]),  # (3)!
...     max_error_interp_mode=vbt.enums.InterpMode.Discrete,
...     min_similarity=0.85
... )
>>> pattern_range_records
array([(0, 0, 270, 314, 1, 0.86226468), (1, 0, 484, 540, 1, 0.89078042)])
```

1. Our price contains only one asset, thus we're using the one-dimensional version of this function
2. If `max_window` is provided, the `window` argument turns into the minimum window
3. Don't forget to wrap any array-like argument with NumPy when dealing with Numba-compiled functions!

The call returned two records, with 86% and 89% (!) similarity respectively.
The first window is `314 - 270 = 44` data points long, the second `540 - 484 = 56`.
Let's plot the second fit by also plotting what happened for 30 bars after the pattern:

```pycon
>>> start_row = pattern_range_records[1]["start_idx"]
>>> end_row = pattern_range_records[1]["end_idx"]
>>> fig = data.iloc[start_row:end_row + 30].plot(plot_volume=False)
>>> price.iloc[start_row:end_row].vbt.plot_pattern(
...     pattern, 
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     plot_obj=False, 
...     fig=fig
... )
```

![](/assets/images/tutorials/patterns/find_pattern.svg)

Voilà, it's the same pattern we processed at the beginning of this tutorial! This example should 
signify how important it is to test a dense grid of windows to find optimal matches. As opposed to the 
[PATSIM](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM) indicator, this approach
consumes almost no memory, and implements a range of tricks to make the calculation faster,
such as pre-calculating the price's expanding minimum and maximum values.

As always, using the raw Numba-compiled function is all fun and games until you meet a 
more convenient method that wraps it: [PatternRanges.from_pattern_search](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.from_pattern_search).
This class method takes all the parameters accepted by [find_pattern_1d_nb](/api/generic/nb/records/#vectorbtpro.generic.nb.records.find_pattern_1d_nb),
builds a grid of parameter combinations, splits the price array into one-dimensional column arrays, 
and executes each parameter combination on each column array using the function 
[execute](/api/utils/execution/#vectorbtpro.utils.execution.execute), which allows for both 
sequential and parallel processing. After processing all the input combinations, the method concatenates
the resulting record arrays, and wraps them with the class [PatternRanges](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges).
This class extends the base class [Ranges](/api/generic/ranges/#vectorbtpro.generic.ranges.Ranges)
with the similarity field and various pattern analysis and plotting methods. Enough theory,
let's do the same as above using this method:

```pycon
>>> pattern_ranges = vbt.PatternRanges.from_pattern_search(
...     price,  # (1)!
...     pattern,
...     window=30,
...     max_window=90,
...     error_type="relative",  # (2)!
...     max_error=0.05,  # (3)!
...     max_error_interp_mode="discrete",
...     min_similarity=0.85
... )
>>> pattern_ranges
<vectorbtpro.generic.ranges.PatternRanges at 0x7f8bdab039d0>
```

1. Accepts Pandas objects
2. Accepts strings
3. Accepts constants

If the price is a Pandas object, there is also an accessor method 
[GenericAccessor.find_pattern](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.find_pattern)
that calls the class method above and can save us a few lines:

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     pattern,
...     window=30,
...     max_window=90,
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     min_similarity=0.85
... )
```

Let's take a look at the records in a human-readable format:

```pycon
>>> pattern_ranges.records_readable
   Pattern Range Id  Column               Start Index  \\
0                 0       0 2021-02-26 00:00:00+00:00   
1                 1       0 2021-09-28 00:00:00+00:00   

                  End Index  Status  Similarity  
0 2021-04-11 00:00:00+00:00  Closed    0.862265  
1 2021-11-23 00:00:00+00:00  Closed    0.890780 
```

Also, the returned [PatternRanges](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges) instance
stores the search configuration that was used to generate those records, per column.
The property [PatternRanges.search_configs](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.search_configs)
returns a list of such configurations, each being an instance of the data class 
[PSC](/api/generic/ranges/#vectorbtpro.generic.ranges.PSC) (stands for "Pattern Search Config").
There is the same number of search configurations as we have columns.

!!! hint
    Take a look at the [documentation of PSC](/api/generic/ranges/#vectorbtpro.generic.ranges.PSC),
    it describes in detail each parameter used in pattern search.

```pycon
>>> pattern_ranges.wrapper.columns
Int64Index([0], dtype='int64')

>>> pattern_ranges.search_configs
[PSC(
    pattern=array([1, 2, 3, 2, 3, 2]), 
    window=30, 
    max_window=120, 
    row_select_prob=1.0, 
    window_select_prob=1.0, 
    roll_forward=False,
    interp_mode=3, 
    rescale_mode=0, 
    vmin=nan, 
    vmax=nan, 
    pmin=nan, 
    pmax=nan, 
    invert=False, 
    error_type=1, 
    distance_measure=0, 
    max_error=array([0.05]), 
    max_error_interp_mode=2, 
    max_error_as_maxdist=False, 
    max_error_strict=False, 
    min_pct_change=nan, 
    max_pct_change=nan, 
    min_similarity=0.85, 
    max_one_per_row=True,
    max_overlap=0, 
    max_records=None, 
    name=None
)]
```

This one configuration instance contains all the argument names and values that were passed 
to [find_pattern_1d_nb](/api/generic/nb/records/#vectorbtpro.generic.nb.records.find_pattern_1d_nb).
Why do we need to keep it? For plotting! Remember how inconvenient it was having to provide to a 
plotting method the exact same arguments that were used in the similarity calculation. To
make things more streamlined, each pattern range instance keeps track of the search configuration
for each column to be plotted. The plotting itself is done with the method [PatternRanges.plot](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.plot),
which uses the method [Ranges.plot](/api/generic/ranges/#vectorbtpro.generic.ranges.Ranges.plot)
for plotting the data and ranges, and the method [GenericSRAccessor.plot_pattern](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericSRAccessor.plot_pattern)
for plotting the patterns:

```pycon
>>> pattern_ranges.plot()
```

![](/assets/images/tutorials/patterns/pattern_ranges.svg)

By default, it fills the distance between the price and the pattern (set `fill_distance=False` to hide) 
and it doesn't display the corridor (set `plot_max_error=True` to show). As any other subclass
of [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable), an instance
of [PatternRanges](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges) behaves in many 
regards like a regular Pandas object. For example, we can filter a date range using the regular
`loc` and `iloc` operations to zoom in into any pattern programmatically.

!!! note
    When selecting a date range, the indexing operation will filter out all range records 
    that do not *completely* fall in the new date range. That is, if a pattern range
    starts on `2020-01-01` and lasts until `2021-01-01`, it will be included in the new
    pattern range instance if the new date range encompasses that period fully, for example
    `2019-01-01` to `2021-01-01`, but not `2019-01-01` to `2020-12-31`.

```pycon
>>> pattern_ranges.loc["2021-09-01":"2022-01-01"].plot()
```

![](/assets/images/tutorials/patterns/pattern_ranges_zoomed.svg)

!!! info
    You might have noticed that the bright area is larger than the window by one bar, but also that the green 
    marker is located further away from the last window point. This is because the field `end_idx`
    represents the excluding end of the range - the first point after the last window point.
    This is needed to calculate the duration of any range properly. The only exception is when the 
    last window point is the last point in the entire price array: in such a case, the marker will be 
    placed at that point and the range will be marked as open. Open ranges don't mean that the pattern
    isn't completed though.

Since the entire information is now represented using records, we can query various useful
metrics to describe the results:

```pycon
>>> pattern_ranges.stats()
Start                 2020-06-01 00:00:00+00:00
End                   2022-05-31 00:00:00+00:00
Period                        730 days 00:00:00
Total Records                                 2
Coverage                               0.136986
Overlap Coverage                            0.0
Duration: Min                  44 days 00:00:00
Duration: Median               50 days 00:00:00
Duration: Max                  56 days 00:00:00
Similarity: Min                        0.862265
Similarity: Median                     0.876523
Similarity: Max                         0.89078
dtype: object
```

Here, for example, we see that there are two non-overlapping patterns covering 13.7% of the 
entire period. We also see various duration and similarity quantiles.

### Overlapping

In the example above, we were searching a total of `90 - 30 + 1 = 61` windows at each single row.
Why haven't we got any overlapping ranges? Because, by default, overlapping is not allowed.
There are two (optional) mechanisms implemented. First, whenever there are multiple windows starting at the
same row, the algorithm will select the one with the highest similarity. This means that the number
of filled records is always guaranteed to be equal or below the number of rows in the price array.
Second, whenever there are multiple consecutive records regardless of whether they start at the same row 
but with overlapping ranges, the algorithm will also select the one with the highest similarity.

But sometimes, there might be a need to permit overlapping ranges; for example, one pattern
might start right before another pattern ends; or, one big pattern might encompass a range of smaller 
patterns. Such scenarios can be addressed by tweaking the argument `overlap_mode` of the type
[OverlapMode](/api/generic/enums/#vectorbtpro.generic.enums.OverlapMode). Setting it to `AllowAll` 
will disable both mechanisms and append every single record. Setting it to `Allow` will disable the 
second mechanism while only filtering those ranges that start at the same row. Setting it to `Disallow` 
will enable both mechanisms (default), while setting it to any other positive integer will treat it as 
the maximum number of rows any two neighboring ranges are allowed to share.

!!! important
    Setting the argument to `AllowAll` may produce a record array that is bigger than the price array. 
    In such a case, you need to manually increase the number of records to be allocated using `max_records`,
    for example, `max_records=len(price) * 2`.

Let's allow overlapping ranges as long as they don't start at the same row:

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     pattern,
...     window=30,
...     max_window=120,
...     error_type="relative",
...     max_error=0.05,
...     max_error_interp_mode="discrete",
...     min_similarity=0.85,
...     overlap_mode="allow"  # (1)!
... )

>>> pattern_ranges.count()
16

>>> pattern_ranges.overlap_coverage
0.9642857142857143
```

1. Here

We see that instead of 2, there are now 16 ranges that have been detected.
Also, ranges overlap by 96%, which means that there are probably no ranges that 
don't share rows with other ranges. Let's visualize the entire thing:

```pycon
>>> pattern_ranges.plot(plot_zones=False, plot_patterns=False)
```

![](/assets/images/tutorials/patterns/pattern_ranges_overlap.svg)

As shown in the graph, there are only two global matches, each being confirmed by windows 
of varying lengths. If we set `overlap_mode="disallow"`, only the most similar windows in each region would remain.

!!! info
    There is an argument that controls the direction in which windows are rolled - `roll_forward`.
    If this argument is `False` (default), ranges will be sorted by the end index and may have multiple
    records pointing to the same start index. Otherwise, ranges will be sorted by the start index and 
    may have multiple records pointing to the same end index. 

### Random selection

Sometimes, not every row and window combination is worth searching. If the input data is too big, 
or there are too many parameter combinations involved, the search would take ages 
to complete in vectorbt's terms (it would still be incredibly fast though!). To make
searchable regions more sparse, we can introduce a probability of picking a certain row/window.
For instance, if the probability is `0.5`, the algorithm would search every second row/window on average.
Let's set a probability of 50% for rows and 25% for windows, and benchmark the execution to see 
whether it would make the execution 8 times faster on average:

```pycon
>>> def run_prob_search(row_select_prob, window_select_prob):
...     return price.vbt.find_pattern(
...         pattern,
...         window=30,
...         max_window=120,
...         row_select_prob=row_select_prob,  # (1)!
...         window_select_prob=window_select_prob,
...         error_type="relative",
...         max_error=0.05,
...         max_error_interp_mode="discrete",
...         min_similarity=0.8,  # (2)!
...     )

>>> %timeit run_prob_search(1.0, 1.0)
111 ms ± 247 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit run_prob_search(0.5, 0.25)
15 ms ± 183 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

1. Both arguments must be from 0 to 1
2. Reduce the similarity threshold to capture more ranges

Just note that, unless you set a random seed (argument `seed`), detected pattern ranges
may vary greatly with each method call. Let's run the function `run_prob_search` 100 times and 
plot the number of filled records:

```pycon
>>> run_prob_search(1.0, 1.0).count()  # (1)!
6

>>> pd.Series([
...     run_prob_search(0.5, 0.25).count() 
...     for i in range(100)
... ]).vbt.plot()
```

1. Get the maximum possible number of detected patterns

![](/assets/images/tutorials/patterns/random_selection.svg)

!!! hint
    The lower are the selection probabilities, the less likely you will detect all patterns
    in a single call, thus always make sure to run the same search multiple times to assess 
    the stability of the detection accuracy.

### Params

The beauty of the class method [PatternRanges.search_configs](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.search_configs)
is in its ability to behave like a full-blown indicator. It uses the same mechanism for broadcasting
and combining parameters as the vectorbt's broadcaster; both are based on the function
[combine_params](/api/utils/params/#vectorbtpro.utils.params.combine_params). To mark any argument
as a parameter, we need to wrap it with [Param](/api/utils/params/#vectorbtpro.utils.params.Param).
This will have several implications: the parameter will be broadcasted and combined with other parameters, 
and it will be reflected as a standalone level in the final column hierarchy.

Let's test 4 patterns: "V-Top", "V-Bottom", a rising market and a falling market pattern.

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     vbt.Param([
...         [1, 2, 1],
...         [2, 1, 2],
...         [1, 2, 3],
...         [3, 2, 1]
...     ]),
...     window=30,
...     max_window=120,
... )
```

[=100% "Param 4/4"]{: .candystripe}

Since we provided more than one parameter combination, the executor displayed a progress bar.
Let's counter the number of found patterns:

```pycon
>>> pattern_ranges.count()
pattern
list_0    3
list_1    0
list_2    7
list_3    3
Name: count, dtype: int64
```

We see that the argument `pattern` received four lists. Let's make it more verbose
by providing an index that gives each list a name:

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     vbt.Param([
...         [1, 2, 1],
...         [2, 1, 2],
...         [1, 2, 3],
...         [3, 2, 1]
...     ], keys=["v-top", "v-bottom", "rising", "falling"]),
...     window=30,
...     max_window=120,
... )
>>> pattern_ranges.count()
pattern
v-top       3
v-bottom    0
rising      7
falling     3
Name: count, dtype: int64
```

Let's display the three detected falling patterns:

```pycon
>>> pattern_ranges.plot(column="falling")
```

![](/assets/images/tutorials/patterns/params_falling.svg)

But there are many more falling patterns on the chart, why haven't they been recognized?
Because 1) the algorithm searched only regions that are at least 30 bars long, 2) the default 
minimum similarity threshold is 85%, such that the algorithm picked only those regions that were the 
most similar to a straight line, and 3) the algorithm removed any overlapping regions.

Now, let's pass multiple parameters. In such a case, their values will be combined to build a 
Cartesian product of all parameter combinations. Let's additionally test multiple similarity thresholds:

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     vbt.Param([
...         [1, 2, 1],
...         [2, 1, 2],
...         [1, 2, 3],
...         [3, 2, 1]
...     ], keys=["v-top", "v-bottom", "rising", "falling"]),
...     window=30,
...     max_window=120,
...     min_similarity=vbt.Param([0.8, 0.85])
... )
>>> pattern_ranges.count()
pattern   min_similarity
v-top     0.80              6
          0.85              3
v-bottom  0.80              3
          0.85              0
rising    0.80              8
          0.85              7
falling   0.80              6
          0.85              3
Name: count, dtype: int64
```

We can finally see some detected "V-Bottom" ranges:

```pycon
>>> pattern_ranges.plot(column=("v-bottom", 0.8))
```

![](/assets/images/tutorials/patterns/params_v_bottom.svg)

What if we didn't want to build a product of some parameters? For instance, what if we wanted
to use different window lengths for different patterns? This is possible by providing a product index.
Parameters that are linked to the same product index are not combined, only broadcasted together.

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     vbt.Param([
...         [1, 2, 1],
...         [2, 1, 2],
...         [1, 2, 3],
...         [3, 2, 1]
...     ], keys=["v-top", "v-bottom", "rising", "falling"], product_idx=0),
...     window=vbt.Param([30, 30, 7, 7], product_idx=0),  # (1)!
...     max_window=vbt.Param([120, 120, 30, 30], product_idx=0),
...     min_similarity=vbt.Param([0.8, 0.85], product_idx=1)  # (2)!
... )
>>> pattern_ranges.count()
pattern   window  max_window  min_similarity
v-top     30      120         0.80               6
                              0.85               3
v-bottom  30      120         0.80               3
                              0.85               0
rising    7       30          0.80              27
                              0.85              23
falling   7       30          0.80              25
                              0.85              15
Name: count, dtype: int64
```

1. Needs to have the same length as other parameters with the same product index
2. Will be combined with "blocks" of parameters sharing the same product index

!!! note
    If used, product index must be provided for all parameters. You can also re-order
    the column levels of some parameters by assigning them a lower/higher product index.

### Configs

It's worth noting that in a case where we have multiple assets, each parameter will be applied on 
the entire price array. But what if we wanted to search for different patterns in the price
of different assets? Remember how each instance of [PatternRanges](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges) 
keeps track of the search configuration of each individual column in 
[PatternRanges.search_configs](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.search_configs)? 
In the same way, we can manually provide search configurations using the argument `search_configs`,
which must be provided as a list of [PSC](/api/generic/ranges/#vectorbtpro.generic.ranges.PSC) instances 
(per entire price), or a list of lists of [PSC](/api/generic/ranges/#vectorbtpro.generic.ranges.PSC) 
instances (per column). This way, we can define arbitrary parameter combinations.

To better illustrate the usage, let's fetch the price of `BTCUSDT` and `ETHUSDT` symbols,
and search for the "Double Top" pattern in both assets, and for any occurrence of the latest 30 bars in
each asset individually:

```pycon
>>> pattern_ranges = mult_price.vbt.find_pattern(
...     search_configs=[
...         vbt.PSC(pattern=[1, 2, 3, 2, 3, 2], window=30),  # (1)!
...         [
...             vbt.PSC(pattern=mult_price.iloc[-30:, 0]),  # (2)!
...             vbt.PSC(pattern=mult_price.iloc[-30:, 1]),
...         ]
...     ],
...     min_similarity=0.8  # (3)!
... )
>>> pattern_ranges.count()
search_config  symbol 
0              BTCUSDT    6
1              ETHUSDT    4
2              BTCUSDT    5
3              ETHUSDT    8
Name: count, dtype: int64
```

1. Every `PSC` instance in the outer list is getting applied per entire array
2. Every `PSC` instance in the inner list is getting applied per column
3. If an argument should be applied to all columns, we can provide it as a regular argument

!!! hint
    We can provide arguments to [PSC](/api/generic/ranges/#vectorbtpro.generic.ranges.PSC) 
    in a human-readable format. Each config will be prepared for the use in Numba automatically.

We see that the column hierarchy now contains two levels: the identifier of the search config,
and the column name. Let's make it more verbose by choosing a name for each config:

```pycon
>>> pattern_ranges = mult_price.vbt.find_pattern(
...     search_configs=[
...         vbt.PSC(pattern=[1, 2, 3, 2, 3, 2], window=30, name="double_top"),
...         [
...             vbt.PSC(pattern=mult_price.iloc[-30:, 0], name="last"),
...             vbt.PSC(pattern=mult_price.iloc[-30:, 1], name="last"),
...         ]
...     ],
...     min_similarity=0.8
... )
>>> pattern_ranges.count()
search_config  symbol 
double_top     BTCUSDT    6
               ETHUSDT    4
last           BTCUSDT    5
               ETHUSDT    8
Name: count, dtype: int64
```

We can also combine search configurations and parameters. In this case, the method will clone
the provided search configurations by the number of parameter combinations, and override the parameters
of each search configuration by the current parameter combination. Let's test various rescaling modes:

```pycon
>>> pattern_ranges = mult_price.vbt.find_pattern(
...     search_configs=[
...         vbt.PSC(pattern=[1, 2, 3, 2, 3, 2], window=30, name="double_top"),
...         [
...             vbt.PSC(pattern=mult_price.iloc[-30:, 0], name="last"),
...             vbt.PSC(pattern=mult_price.iloc[-30:, 1], name="last"),
...         ]
...     ],
...     rescale_mode=vbt.Param(["minmax", "rebase"]),
...     min_similarity=0.8,
...     open=mult_data.open,  # (1)!
...     high=mult_data.high,
...     low=mult_data.low,
...     close=mult_data.close,
... )
>>> pattern_ranges.count()
rescale_mode  search_config  symbol 
minmax        double_top     BTCUSDT    6
                             ETHUSDT    4
              last           BTCUSDT    5
                             ETHUSDT    8
rebase        double_top     BTCUSDT    0
                             ETHUSDT    0
              last           BTCUSDT    2
                             ETHUSDT    2
Name: count, dtype: int64
```

1. Provide OHLC for plotting

For example, our search for a pattern based on the 30 last bars in `ETHUSDT` found 8 occurrences 
similar by shape (min-max rescaling) and only 2 occurrences similar by shape and percentage change (rebasing):

```pycon
>>> pattern_ranges.plot(column=("rebase", "last", "ETHUSDT"))
```

![](/assets/images/tutorials/patterns/configs_and_params.svg)

The first range has the similarity of 85%, while the second range is still open and has the similarity 
of 100%, which makes sense because it was used as a pattern.

!!! note
    Again, open range doesn't mean that it hasn't finished developing - it only means that the last
    point in the range is the last point in the price array such that the duration can be calculated correctly.

### Mask

So, how do we use a pattern range instance to generate signals? Since such an instance usually
stores only ranges that passed a certain similarity threshold, we only need to know whether 
there is any range that closes at a particular row and column. Such a mask can be generated by
calling the property [PatternRanges.last_pd_mask](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.last_pd_mask):

```pycon
>>> mask = pattern_ranges.last_pd_mask
>>> mask.sum()  # (1)!
rescale_mode  search_config  symbol 
minmax        touble_top     BTCUSDT    6
                             ETHUSDT    4
              last           BTCUSDT    5
                             ETHUSDT    8
rebase        touble_top     BTCUSDT    0
                             ETHUSDT    0
              last           BTCUSDT    2
                             ETHUSDT    2
dtype: int64
```

1. Count `True` values in the mask

We can then use this mask, for example, in [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals).

### Indicator

If we don't care about plotting and analyzing pattern ranges, we can use the same 
[PATSIM](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM) indicator that we used 
previously to generate a Series/DataFrame with similarity scores. What we didn't discuss previously though
is that this indicator also takes the arguments `max_window`, `row_select_prob`, and `window_select_prob`. 
Let's prove that the indicator produces the same similarity scores as pattern ranges:

```pycon
>>> pattern_ranges = price.vbt.find_pattern(
...     pattern,
...     window=30,
...     max_window=120,
...     row_select_prob=0.5,
...     window_select_prob=0.5,
...     overlap_mode="allow",  # (1)!
...     seed=42
... )
>>> pr_mask = pattern_ranges.map_field(
...     "similarity", 
...     idx_arr=pattern_ranges.last_idx.values
... ).to_pd()
>>> pr_mask[~pr_mask.isnull()]
Open time
2021-03-23 00:00:00+00:00    0.854189
2021-03-26 00:00:00+00:00    0.853817
2021-04-10 00:00:00+00:00    0.866913
2021-04-11 00:00:00+00:00    0.866106
2021-11-17 00:00:00+00:00    0.868276
2021-11-18 00:00:00+00:00    0.873757
2021-11-21 00:00:00+00:00    0.890225
2021-11-23 00:00:00+00:00    0.892541
2021-11-24 00:00:00+00:00    0.879475
2021-11-26 00:00:00+00:00    0.877245
2021-11-27 00:00:00+00:00    0.872172
dtype: float64

>>> patsim = vbt.PATSIM.run(
...     price,
...     vbt.Default(pattern),  # (2)!
...     window=vbt.Default(30),
...     max_window=vbt.Default(120),
...     row_select_prob=vbt.Default(0.5),
...     window_select_prob=vbt.Default(0.5),
...     min_similarity=vbt.Default(0.85),  # (3)!
...     seed=42
... )
>>> ind_mask = patsim.sim
>>> ind_mask[~ind_mask.isnull()]
Open time
2021-03-23 00:00:00+00:00    0.854189
2021-03-26 00:00:00+00:00    0.853817
2021-04-10 00:00:00+00:00    0.866913
2021-04-11 00:00:00+00:00    0.866106
2021-11-17 00:00:00+00:00    0.868276
2021-11-18 00:00:00+00:00    0.873757
2021-11-21 00:00:00+00:00    0.890225
2021-11-23 00:00:00+00:00    0.892541
2021-11-24 00:00:00+00:00    0.879475
2021-11-26 00:00:00+00:00    0.877245
2021-11-27 00:00:00+00:00    0.872172
dtype: float64
```

1. This is the only mode supported in [PATSIM](/api/indicators/custom/#vectorbtpro.indicators.custom.PATSIM)
2. Make each argument default to hide it from columns
3. This is the default threshold in [PatternRanges.from_pattern_search](/api/generic/ranges/#vectorbtpro.generic.ranges.PatternRanges.from_pattern_search)

## Combination

We know how to generate signals from one pattern found in one array, but what about a use case where 
our signals should only be triggered upon a combination of patterns found across different arrays? 
For example, how do we quantify convergence and divergence? To combine multiple patterns conditionally,
we need to combine their similarity scores. For example, below we're searching for a bearish divergence 
between the high price and MACD:

```pycon
>>> price_highs = vbt.PATSIM.run(
...     data.high, 
...     pattern=np.array([1, 3, 2, 4]),  # (1)!
...     window=40,
...     max_window=50
... )
>>> macd = data.run("talib_macd").macd  # (2)!
>>> macd_lows = vbt.PATSIM.run(
...     macd, 
...     pattern=np.array([4, 2, 3, 1]),  # (3)!
...     window=40,
...     max_window=50
... )

>>> fig = vbt.make_subplots(
...     rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
... )
>>> fig.update_layout(height=500)
>>> data.high.rename("Price").vbt.plot(
...     add_trace_kwargs=dict(row=1, col=1), fig=fig
... )
>>> macd.rename("MACD").vbt.plot(
...     add_trace_kwargs=dict(row=2, col=1), fig=fig
... )
>>> price_highs.sim.rename("Price Sim").vbt.plot(
...     add_trace_kwargs=dict(row=3, col=1), fig=fig
... )
>>> macd_lows.sim.rename("MACD Sim").vbt.plot(
...     add_trace_kwargs=dict(row=3, col=1), fig=fig
... )
>>> fig
```

1. Two rising highs
2. Use [Data.run](/api/data/base/#vectorbtpro.data.base.Data.run) to quickly run indicators
3. Two falling highs

![](/assets/images/tutorials/patterns/macd_divergence.svg)

Upon looking at the chart, we can confirm that the selected parameters accurately represent
the events we are looking for - the regions where the price and MACD is rising and falling respectively
also leads to the rise of their similarity score. We also can derive the optimal similarity threshold
that would yield a moderate amount of crossovers - 80%. In addition, the points where the price's
similarity line crosses the threshold often happens slightly before the MACD's, thus it's insufficient
to simply test whether both crossovers happen at the same time - we need to introduce a waiting time
using the rolling "any" operation. Below, for example, we get an exit signal if both similarities crossed 
the threshold during the last 10 bars, and not necessarily at the same time:

```pycon
>>> cond1 = (price_highs.sim >= 0.8).vbt.rolling_any(10)
>>> cond2 = (macd_lows.sim >= 0.8).vbt.rolling_any(10)
>>> exits = cond1 & cond2
>>> fig = data.plot(ohlc_trace_kwargs=dict(opacity=0.5))
>>> exits.vbt.signals.plot_as_exits(data.close, fig=fig).show_svg()
```

![](/assets/images/tutorials/patterns/macd_divergence_exits.svg)

Fore more ideas, take a look into [Signal Development](/tutorials/signal-development).
