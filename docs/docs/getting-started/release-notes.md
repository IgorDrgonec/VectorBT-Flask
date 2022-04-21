---
title: Release notes
---

# Release notes

All notable changes in reverse chronological order.

## Version 1.2.2 (21 Apr, 2022)

- Made the call sequence array entirely optional. Prior to this change, the user had to create an array 
of the same size as the target shape, which is pretty unnecessary and consumes lots of memory. Now, 
setting `call_seq` to None or `auto` won't require a user-defined array anymore, and only the call 
sequence for the current row is being kept in memory.
- The close price in [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)
won't broadcast to the full shape anymore. Since all methods deploy flexible indexing, 
there is no need to expand and materialize the close price. Instead, it keeps its original shape while 
the broadcasting operation returns a wrapper that holds the target shape and other Pandas metadata. 
This has one big advantage: lower memory footprint.
- Disabled `flex_2d` everywhere for more consistency across the codebase. Passing one-dimensional
arrays will treat them per-row by default.
- Fixed the shape in the wrapper returned by [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast)
- When specifying enum fields such as "targetamount", the mapper will ignore all non-alphanumeric 
characters by default, thus "Target Amount" can now be passed as well
- Added the option `incl_doc` to show/hide the docstring of the function when using 
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func)
- [ArrayWrapper.wrap](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.wrap)
will broadcast the to-be-wrapped array using NumPy rules if its shape doesn't match the shape of the wrapper.
This way, smaller arrays can expand to the target shape once this is really required - good for memory.
- Added the option `indexer_method` to specify the indexer method in 
[BaseAccessor.set](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set) 
and [BaseAccessor.set_between](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set_between)
to control which timestamp (previous or next?) should be used if there is no exact match
- Cash deposits and cash earnings now respect the cash balance in 
[Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) 
and [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals).
But also, cash deposits now behave the same way as cash earnings: a full array will be created during
the simulation if any element of the passed cash deposits is not zero; this array will be overridden 
in-place at each row and group, and then returned as a part of the simulation output.
- Added the option `skipna` in [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders)
for skipping rows where the size in all columns of a group is NaN
- Wrote documentation on [From orders](/documentation/portfolio/from-orders/) :notebook_with_decorative_cover:

## Version 1.2.1 (10 Apr, 2022)

- Fixed check for zero size in [buy_nb](/api/portfolio/nb/core/#vectorbtpro.portfolio.nb.core.buy_nb)
- Fixed [is_numba_func](/api/utils/checks/#vectorbtpro.utils.checks.is_numba_func) when Numba is disabled globally
- Use record counts (`order_counts` and `log_counts`) instead of last indices (`last_oidx` and `last_lidx`) 
in simulation methods with order functions
- Added `inplace` argument for [BaseAccessor.set](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set) 
and [BaseAccessor.set_between](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set_between) 
- Fixed `ndim` in [Data.get_symbol_wrapper](/api/data/base/#vectorbtpro.data.base.Data.get_symbol_wrapper)
- Renamed `ExecuteOrderState` to [AccountState](/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccountState) 
and `ProcessOrderState` to [ExecState](/api/portfolio/enums/#vectorbtpro.portfolio.enums.ExecState) 
- Rotational indexing is disabled by default and can be enabled globally using settings
- Order approximation function takes an instance of `ExecState` instead of state variables
- Created classes [RandomOHLCData](/api/data/custom/#vectorbtpro.data.custom.RandomData) and
  [GBMOHLCData](/api/data/custom/#vectorbtpro.data.custom.GBMOHLCData) for generation of random OHLC data
- Made numeric tests more precise by reducing tolerance values
- Fixed documentation where argument `interval` was used instead of `timeframe`
- Wrote documentation on [Portfolio simulation](/documentation/portfolio/) :notebook_with_decorative_cover:

## Version 1.2.0 (3 Apr, 2022)

- Integrated [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/): implemented
the function [pypfopt_optimize](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.pypfopt_optimize)
and an infrastructure around it to run arbitrary optimization models from PyPortfolioOpt with a single function call.
- Split generation and reduction of resampling metadata. The generation part now resides in
[resampling.nb](/api/base/resampling/nb/), while the reduction part takes place in 
[generic.nb.apply_reduce](/api/generic/nb/apply_reduce/).
- Implemented wrapper methods for generation of index points ([ArrayWrapper.get_index_points](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.get_index_points)) 
and index ranges ([ArrayWrapper.get_index_ranges](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.get_index_ranges)) 
from human-readable queries. This helps tremendously in rebalancing.
- Implemented the [PortfolioOptimizer](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer) class,
which is capable of portfolio optimization at regular and irregular intervals, storing the generated allocation data
in a compressed format, and analyzing and plotting it. Supports the following input modes:
    1. Custom allocation function ([PortfolioOptimizer.from_allocate_func](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_allocate_func))
    2. Custom optimization function ([PortfolioOptimizer.from_optimize_func](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_optimize_func))
    3. Custom allocations ([PortfolioOptimizer.from_allocations](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_allocations))
    4. Custom filled allocations ([PortfolioOptimizer.from_filled_allocations](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_filled_allocations))
    5. Random allocation ([PortfolioOptimizer.from_random](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_random))
    6. Uniform allocation ([PortfolioOptimizer.from_uniform](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_uniform))
    7. PyPortfolioOpt ([PortfolioOptimizer.from_pypfopt](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_pypfopt))
    8. Universal Portfolios ([PortfolioOptimizer.from_universal_algo](/api/portfolio/pfopt/base/#vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer.from_universal_algo))
- Added time frames as a parameter to TA-Lib indicators. This will downsample the input data 
(such as the close price), run the indicator, and upsample it back to the original time frame.
Multiple time frame combinations are supported out of the box.
- Implemented convenience methods [BaseAccessor.set](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set) 
and [BaseAccessor.set_between](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.set_between) 
for setting data based on index points and ranges respectively
- Added an attribute class [ExceptLevel](/api/base/grouping/#vectorbtpro.base.grouping.base.ExceptLevel)
that can be used to specify the level by which **not** to group in `group_by`. This is handy
when there are many levels and there is a need to group by all levels except assets, for example.
- Wrote [Portfolio optimization](/tutorials/portfolio-optimization) :notebook_with_decorative_cover:

## Version 1.1.2 (12 Mar, 2022)

- Added option `skipna` to run a TA-Lib indicator on non-NA values only (TA-Lib hates NaN)
- Implemented a range of (NB and SP) resampling functions for
    1. Getting the latest information at each timestamp, supporting bar data ([GenericAccessor.latest_at_index](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.latest_at_index))
    2. Resampling to a custom index, both as a regular and meta method ([GenericAccessor.resample_to_index](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_to_index))
    3. Resampling between custom bounds, both as a regular and meta method ([GenericAccessor.resample_between_bounds](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_between_bounds))
- Implemented the [Resampler](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler) class, 
which acts as a mapper between the source and target index for best flexibility. It can parse a resampler
from Pandas. Also, implemented a range of helper functions for
    1. Generating a datetime index from frequency (NB)
    2. Getting the right bound of a datetime index
    3. Mapping one datetime index to another (NB, [Resampler.map_to_target_index](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler.map_to_target_index))
    4. Getting datetime index difference (NB, [Resampler.index_difference](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler.index_difference))
- Implemented an interface for resampling complex vectorbt objects in form of an abstract method
[Wrapping.resample](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping.resample). Also, defined
resampling logic for all classes whose Pandas objects can be resampled:
    1. [Data](/api/data/base/#vectorbtpro.data.base.Data)
    2. [OHLCVDFAccessor](/api/ohlcv/accessors/#vectorbtpro.ohlcv.accessors.OHLCVDFAccessor)
    3. [ReturnsAccessor](/api/returns/accessors/#vectorbtpro.returns.accessors.ReturnsAccessor)
    4. [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)
    5. [Records](/api/records/base/#vectorbtpro.records.base.Records) and all its subclasses
- Introduced the column config [Data.column_config](/api/data/base/#vectorbtpro.data.base.Data.column_config),
which can be used to define resampling function for each custom column in [Data](/api/data/base/#vectorbtpro.data.base.Data).
OHLCV data is resampled automatically.
- Completely refactored handling of in-outputs in [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio).
Introduced the in-output config [Portfolio.in_output_config](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.in_output_config),
which can be used to define the layout, array type, and resampling function for each custom in-output.
Appending suffixes (such as `_pcg`) to in-output names is now optional. The suffix `_pcgs` has been renamed to `_cs`.
The resolution mechanism for in-outputs has been distributed over multiple class methods and made more transparent.
- The in-output name for returns has been changed from `returns_pcgs` to just `returns`
- Added argument `init_price` to specify the original entry price of `init_position`. This makes calculation
of P&L and other metrics more precise and flexible.
- Fixed the issue where only the first row in `cash_deposits` was applied in 
[Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) 
and [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
- Distributed generic Numba-compiled functions across multiple files
- Enabled passing additional keyword arguments to `get_klines` in [BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData)
- Wrote [MTF analysis](/tutorials/mtf-analysis) :notebook_with_decorative_cover:

## Version 1.1.1 (25 Feb, 2022)

- [Data](/api/data/base/#vectorbtpro.data.base.Data) now removes duplicates in index while keeping only the last entry
- Added option `concat` to [Data.update](/api/data/base/#vectorbtpro.data.base.Data.update) for being
able to disable concatenation of new data with existing data and to only return new data
- Added support for chunking in [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) and
[HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) using `chunk_func`
- Implemented class [DataSaver](/api/data/saver/#vectorbtpro.data.saver.DataSaver) that can periodically
collect and save data to disk, and two its subclasses: [CSVDataSaver](/api/data/saver/#vectorbtpro.data.saver.CSVDataSaver)
for writing to CSV files and [HDFDataSaver](/api/data/saver/#vectorbtpro.data.saver.HDFDataSaver) 
for writing to HDF files. This way, one can gradually collect and persist any data from any data provider!
- Fixed unpickling of [Config](/api/utils/config/#vectorbtpro.utils.config.Config) 
- Fixed scheduling in [DataUpdater](/api/data/updater/#vectorbtpro.data.updater.DataUpdater) such that
repeatedly stopping and starting the same updater won't trigger the same job more than once
- [QSAdapter](/api/returns/qs_adapter/#vectorbtpro.returns.qs_adapter.QSAdapter) will now remove
timezone info automatically to prevent issues in QuantStats
- Implemented three new data classes:
    1. [PolygonData](/api/data/custom/#vectorbtpro.data.custom.PolygonData) for [Polygon.io](https://polygon.io/).
Can load data of any size in bunches and respects the API rate limits.
    2. [AlphaVantageData](/api/data/custom/#vectorbtpro.data.custom.AlphaVantageData) for [Alpha Vantage](https://www.alphavantage.co/).
Does not use the `alpha_vantage` library, which isn't actively developed. Instead, it implements a parser
of the Alpha Vantage's documentation website and handles communication with the API using the parsed metadata.
This enables instant reaction to any changes in the Alpha Vantage's API. The user can still disable the parsing 
and specify every bit of information manually.
    3. [NDLData](/api/data/custom/#vectorbtpro.data.custom.NDLData) for [Nasdaq Data Link](https://data.nasdaq.com/).
Supports (time-series) datasets. 
- Fixed the condition for a backward fill in [fbfill_nb](/api/generic/nb/#vectorbtpro.generic.nb.base.fbfill_nb)
- Fixed passing `execute_kwargs` in [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_apply_func).
Also, whenever `apply_func` is Numba compiled, makes the parameter selection function Numba compiled
as well (`jit_select_params=True`) and also releases the GIL (`nogil=True`), so the indicator can be used 
in multithreading right away.


## Version 1.1.0 (20 Feb, 2022)

- Removed support for Python 3.6. Wanted to add support for Python 3.10 but stumbled upon issues
that can be followed in numba/numba#7812 and numba/numba#7839.
- Removed [Bottleneck](https://github.com/pydata/bottleneck) from optional requirements as it 
causes installation issues, but it can still be installed manually
- Formatted the entire codebase with [Black](https://black.readthedocs.io/en/stable/) :black_large_square:
- One or more symbols can be selected from a [Data](/api/data/base/#vectorbtpro.data.base.Data) instance
- Refactored [Data.fetch_kwargs](/api/data/base/#vectorbtpro.data.base.Data.fetch_kwargs) into a symbol dictionary.
Getting the keyword arguments used for fetching a specific symbol is now as simple as `fetch_kwargs[symbol]`.
- Implemented the method [Data.merge](/api/data/base/#vectorbtpro.data.base.Data.merge) for merging
multiple [Data](/api/data/base/#vectorbtpro.data.base.Data) instances
- One or more symbols can be renamed in a [Data](/api/data/base/#vectorbtpro.data.base.Data) instance
- Improved aggregation of metrics in [StatsBuilderMixin](/api/generic/stats_builder/#vectorbtpro.generic.stats_builder.StatsBuilderMixin)
- Removed `ohlc` accessor to avoid confusion. The only accessor is now `ohlcv`.
- [Data.plot](/api/data/base/#vectorbtpro.data.base.Data.plot) automatically plots the data as a candlestick
chart if it can find the right price columns, otherwise, it plots each column as a separate line 
(as it was previously). Also, one can now select a symbol to plot using the `symbol` argument.
- One metric/subplot can be expanded into multiple metrics/subplots using templates. This enables
displaying a variable number of metrics/subplots.
- [Data.plots](/api/data/base/#vectorbtpro.data.base.Data.plots) now plots one subplot per symbol
- Delimiter is recognized automatically when dealing with CSV and TSV files
- Implemented a function for pretty-printing directory trees - [tree](/api/utils/path_/#vectorbtpro.utils.path_.tree)
- Refactored the path matching mechanism for CSV and HDF files. In particular:
    - Path unfolding has been renamed to path matching
    - Wildcards (`*`) are now supported for groups and keys in HDF files
    - Paths can be further filtered using a regex pattern `match_regex`
    - Functions for path matching have become class methods for seamless inheritance
    - The argument `parse_paths` in [LocalData.fetch](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch) has been renamed to `match_paths`
    - The argument `path` in [LocalData.fetch](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch) has been renamed to `paths`
- Symbols that return `None` or an empty array are skipped. When `raise_on_error` is True, any
symbol raising an error is skipped as well.
- Similar to the Python's `help` command, vectorbt now has a function 
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func) that 
pretty-prints the arguments and docstring of any function. It's main advantage is the ability to skip 
annotations, which sometimes reduce readability when exploring more complex vectorbt functions using `help`.
- Settings have been refactored once again: it's now clearly visible which key can be accessed
via the dot notation (_hint_: it must be of type [ChildDict](/api/utils/config/#vectorbtpro.utils.config.ChildDict)).
Also, the argument `convert_dicts_` in [Config](/api/utils/config/#vectorbtpro.utils.config.Config) 
has been renamed to `convert_children_`.
- Moved default values of various [Data](/api/data/base/#vectorbtpro.data.base.Data) classes from the 
[custom](/api/data/custom/) module to [_settings.data](/api/_settings/#vectorbtpro._settings.data) 
to be able to set them globally
- Minor fixes and enhancements across the project
- Wrote documentation on [Data](/documentation/data/) :notebook_with_decorative_cover:

## Version 1.0.10 (11 Feb, 2022)

- Moved the `per_column` logic from [run_pipeline](/api/indicators/factory/#vectorbtpro.indicators.factory.run_pipeline) 
to the indicator function (`custom_func`). Previously, the indicator function received only one column 
and parameter combination at a time, which created issues for caching. Now, the pipeline passes all 
columns and parameter combinations, so it's the responsibility of the indicator function to distribute 
the combinations properly (no worries, `apply_func` will handle it automatically).
- Apply functions (`apply_func`) can run on one-dimensional input data just as well as on two-dimensional 
input data by passing `takes_1d=True` to [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_apply_func).
This mode splits each input array (both Pandas and NumPy) into columns and builds a product of columns
and parameter combinations. Benchmarks show that this has no real implications on performance +
functions that process one column at a time are much easier to write.
- `@talib` and `@talib_1d` annotations were merged into a single `@talib` annotation that 
can handle both one and two-dimensional input data
- Removed automatic module search when parsing indicator expressions, which degraded performance. It's now
recommended to use multi-line expressions and `import` statements.
- Renamed `kwargs_to_args` to `kwargs_as_args` in [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_apply_func)
- Refactored the [execute](/api/utils/execution/#vectorbtpro.utils.execution.execute) function to enable
chunking in indicators
- Fixed the calculation of crossovers. Previously, it ignored a crossover if there was another crossover
one tick behind.
- Greatly optimized stacking of output arrays with one column after running an indicator
- Refactored the `as_attrs_` behavior in [Config](/api/utils/config/#vectorbtpro.utils.config.Config).
Keys won't be attached to the config instance anymore but managed dynamically (= less side effects
when pickling and unpickling).
- Implemented a wide range of inputs states, output states, and accumulators for the use in 
streaming functions, such as in [rolling_mean_1d_nb](/api/generic/nb/#vectorbtpro.generic.nb.rolling.rolling_mean_1d_nb)
- Made flexible indexing with [flex_select_auto_nb](/api/base/indexing/#vectorbtpro.base.indexing.flex_select_auto_nb) 
rotational. For example, if there is a smaller array with 3 columns and a bigger one with 6 columns,
there is no need to tile the smaller array 2 times to match the bigger one - we can simply rotate 
over the smaller array.
- Added support for short names in indicator expressions
- Returns can be pre-computed in both [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) 
and [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals)
by passing the `fill_returns=True` flag
- Fixed [save_animation](/api/utils/image_/#vectorbtpro.utils.image_.save_animation), which previously
produced one less iteration
- Wrote [SuperFast SuperTrend](/tutorials/superfast-supertrend) :notebook_with_decorative_cover:

## Version 1.0.9 (2 Feb, 2022)

- Upgraded the parser of indicator expressions:
    - Can evaluate the expression using [pandas.eval](https://pandas.pydata.org/docs/reference/api/pandas.eval.html)
    - Special variables and commands are annotated with the prefix `@`
    - Supports single as well as multi-line expressions
    - Context variables with the same name aren't needlessly re-computed anymore
    - Can parse the class name at the beginning of the expression
    - Supports one and two-dimensional TA-Lib indicators out-of-the-box
    - Supports magnet names for inputs, in-outputs, and parameters
    - Can parse settings from within the expression and pass them to [IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr)
    - Can automatically resolve annotated indicators
- Enabled plotting of any TA-Lib indicator programmatically by parsing its output flags
- Implemented Numba-compiled functions for Wilder's EMA and STD (SP)
- Renamed `from_custom_func` to `with_custom_func` and `from_apply_func` to `with_apply_func`
since those are instance methods
- Set `minp` (minimum periods) to `None` in generic rolling functions to make the values of 
incomplete windows NaN
- Made parameter selection optional in [IndicatorFactory.with_apply_func](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_apply_func).
Setting `select_params` to False will prepend the iteration index to the arguments of the apply function.
- Refactored the `keep_pd=True` flag to avoid back-and-forth conversion between broadcasted Pandas objects 
and NumPy arrays. Pandas objects are now directly forwarded to the custom function without any pre-processing.
- Renamed `custom_output_props` to `lazy_outputs`
- Renamed `select_one` and `select_one_from_obj` to `select_col` and `select_col_from_obj` respectively
in [Wrapping](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping)
- Input arrays passed to TA-Lib indicators are converted to the data type `np.double`
- Renamed `mapping` to `context` when it comes to templates
- Updated plotting methods in [custom](/api/indicators/custom/)
- [Data.get](/api/data/base/#vectorbtpro.data.base.Data.get) also accepts symbol(s)
- Greatly optimized [Data.concat](/api/data/base/#vectorbtpro.data.base.Data.concat)
- Wrote documentation on [Indicators](/documentation/indicators/) :notebook_with_decorative_cover:

## Version 1.0.8 (25 Jan, 2022)

- Implemented the following Numba-compiled functions:
    - Ranking and rolling ranking (SP)
    - Covariance and rolling covariance (SP)
    - Rolling sum and product (SP)
    - Rolling weighted average (SP)
    - Rolling argmin and argmax
    - Demeaning within a group
- Implemented volume-weighted average price (VWAP)
- Built a powerful engine for parsing indicator expressions. Using 
[IndicatorFactory.from_expr](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_expr),
an indicator can be automatically constructed out of an expression string, such as 
`"rolling_mean((low + high) / 2, 10)"`! [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory)
can recognize common inputs such as `low` and the number of outputs. For parameters and more cryptic inputs,
the user can provide a prefix: `"rolling_mean(abs(in_ts), p_window)"`. Moreover, apart from preset functions
such as `rolling_mean`, whenever it recognizes an unknown function, searches for its implementation 
in various parts of vectorbt and NumPy. Supports NumExpr to accelerate simpler expressions.
- Translated each one of [WorldQuant's 101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) 
into an indicator expression and implemented a convenience method 
[IndicatorFactory.from_wqa101](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory.from_wqa101)
(also as a shortcut `vbt.wqa101`) for executing them
- Unified column grouping for matrices and records across the entire codebase.
Most logic now resides in the sub-package [grouping](/api/base/grouping/). Also, most functions 
(apart from the portfolio-related ones) do not require strict group ordering anymore.
- Added chunking specification for labeling functions
- Fixed [Config.prettify](/api/utils/config/#vectorbtpro.utils.config.Config.prettify) for non-string keys
- Wrote [Basic RSI strategy](/tutorials/basic-rsi-strategy) :notebook_with_decorative_cover:

## Version 1.0.7 (16 Jan, 2021)

- Changed `np.int_` to `np.integer` when passed to `np.issubdtype`
- Refactored auto-aligned initial cash to be based on free cash flows and cash deposits
- Upgraded the string parser of [deep_getattr](/api/utils/attr_/#vectorbtpro.utils.attr_.deep_getattr) 
to accept strings of chained method calls (more intuitive)
- Implemented superfast Pearson correlation coefficient and its rolling version
- Created the class [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable),
which combines [Wrapping](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping) and builder mixins
- Metrics and subplots that require a single column won't raise an error if the object
is two-dimensional and has only one column
- [Grouper](/api/base/grouping/#vectorbtpro.base.grouping.base.Grouper) can return a group map,
which isn't tied to a strict group ordering and is easier to use outside of Numba
- Wrote documentation on [Building blocks](/documentation/building-blocks/) :notebook_with_decorative_cover:

## Version 1.0.6 (9 Jan, 2022)

- Benchmark can be disabled by passing `bm_close=False` to 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)
- Wrote documentation on [Fundamentals](/documentation/fundamentals/) :notebook_with_decorative_cover:

## Version 1.0.5 (8 Jan, 2022)

- Fixed [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
for `direction='both'` and `size_type='value'`. Previously, the position couldn't be properly reversed.
- Avoid sorting paths in [LocalData](/api/data/custom/#vectorbtpro.data.custom.LocalData) 
if they are passed as a sequence

## Version 1.0.4 (6 Jan, 2022)

- Set benchmark easily and also globally (#32) by passing `bm_close` to 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio).
Works similarly to [Portfolio.close](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.close).
- Shortened registry names (such as from `ca_registry` to `ca_reg`)

## Version 1.0.3 (5 Jan, 2022)

- Automatic discovery of symbols for local data (#27): No more need to specify a path to each
CSV/HDF file or HDF key. Passing a path to a directory will traverse each file in this directory.
Passing a [glob-style pattern](https://en.wikipedia.org/wiki/Glob_(programming)) will use `glob.glob`
to traverse all files that match this pattern. Passing an HDF file will extract all keys inside this file.
All the options above can be combined in arbitrary ways.
- Minor fixes in the formatting module
- Removed strict ordering of group and shape suffixes in 
[Portfolio.get_in_output](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.get_in_output) 
and [Portfolio.in_outputs_indexing_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.in_outputs_indexing_func). 
For instance, there is no more difference between in-outputs with names `myinout_2d_pc` and `myinout_pc_2d`.
- Improved readability of cacheable setups: When displaying the status overview, objects are represented
by shorter, human-readable strings, and also contain the position of this object in all the objects 
registered globally and sorted by time. For instance, `portfolio:2` means that there exist 2 more 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) objects created earlier than 
this object. This makes it easier to manage memory and to debug garbage collection.
- Wrapping in-outputs automatically: Implements the method 
[Portfolio.get_in_output](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.get_in_output), 
which can be used to access and automatically wrap any in-output object from 
[Portfolio.in_outputs](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.in_outputs).
- All registries can be accessed directly via `vbt`
- Updates and minor fixes to the ca_registry and ch_registry modules
- Disabled parallelization of [generate_both](/api/signals/accessors/#vectorbtpro.signals.accessors.SignalsAccessor.generate_both), 
which behaved unexpectedly
- Fixed `in_outputs` in [Portfolio.from_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_order_func) 
to work with Numba 0.53.1. This requires the user to define `in_outputs` as a named tuple prior 
to passing to the method.
- Added mapping of `window` to `rolling_period` in the QuantStats adapter

## Version 1.0.2 (31 Dec, 2021)

- Added Alpaca data source (#31). In contrast to the community version, additionally 
allows passing a pre-configured REST object to the 
[AlpacaData.fetch](/api/data/custom/#vectorbtpro.data.custom.AlpacaData.fetch) method.
- Changed the default index field of 
[EntryTrades](/api/portfolio/trades/#vectorbtpro.portfolio.trades.EntryTrades) from `exit_idx` to `entry_idx`
- Dropped JSON and implemented a custom formatting engine that represents objects in Python format.
This perfectly aligns with the switch to dataclasses vectorbt PRO has made. Here's a comparison of 
a wrapper being printed out by the community version and JSON, and vectorbt PRO with the new engine:

```plaintext
ArrayWrapper(**Config({
    "index": "<RangeIndex at 0x1045815e8> of shape (3,)",
    "columns": "<Int64Index at 0x1045815e8> of shape (1,)",
    "ndim": 1,
    "freq": null,
    "column_only_select": null,
    "group_select": null,
    "grouped_ndim": null,
    "group_by": null,
    "allow_enable": true,
    "allow_disable": true,
    "allow_modify": true
}))
```

```python
ArrayWrapper(
    index=<RangeIndex at 0x1045815e8 of shape (3,)>,
    columns=<Int64Index at 0x1045815e8 of shape (1,)>,
    ndim=1,
    freq=None,
    column_only_select=None,
    group_select=None,
    grouped_ndim=None,
    grouper=Grouper(
        index=<Int64Index at 0x1045815e8 of shape (1,)>,
        group_by=None,
        allow_enable=True,
        allow_disable=True,
        allow_modify=True
    )
)
```

## Version 1.0.1 (21 Dec, 2021)

- Adapted the codebase to the new documentation format
- Upgraded the documentation website generator from pdoc3 to MkDocs (Material Insiders). 
API is being automatically converted to Markdown files by a modified version of pdoc3 that 
resides in a private repository of @polakowo.

## Version 1.0.0 (13 Dec, 2021)

!!! info
    This section briefly describes major changes made to the community version. For more details, see commits.

### Execution

- Parallelized most functions that take 2-dimensional arrays using [Explicit Parallel Loops](https://numba.pydata.org/numba-doc/0.37.0/user/parallel.html#explicit-parallel-loops)
- Built an infrastructure for chunking. Any Python function can be wrapped with the
[chunked](/api/utils/chunking/#vectorbtpro.utils.chunking.chunked) decorator, which returns a new 
function with the identical signature but capable of 1) splitting passed positional and keyword arguments 
into multiple chunks, 2) executing each chunk of arguments using the wrapped function, and 3) merging back 
the results. The rules by which the arguments are split and the results are merged must be explicitly 
provided using `arg_take_spec` and `merge_func` respectively. The chunk taking and merging specification 
is provided to most of the Numba-compiled functions that take 2-dimensional arrays. To only chunk functions 
by request, the decorator [register_chunkable](/api/registries/ch_registry/#vectorbtpro.registries.ch_registry.register_chunkable) 
was created, which leaves the Python function unwrapped and registers a so-called "setup" with all
specifications by the global registry
[ChunkableRegistry](/api/registries/ch_registry/#vectorbtpro.registries.ch_registry.ChunkableRegistry). 
Additionally, there are multiple present engines for executing chunks:
[SequenceEngine](/api/utils/execution/#vectorbtpro.utils.execution.SequenceEngine) (a simple queue), 
[DaskEngine](/api/utils/execution/#vectorbtpro.utils.execution.DaskEngine) (mainly for multithreading), 
and [RayEngine](/api/utils/execution/#vectorbtpro.utils.execution.RayEngine) (mainly for multiprocessing).
- Built an infrastructure for wrapping and running JIT-able functions. At the heart of it is the 
[register_jitted](/api/registries/jit_registry/#vectorbtpro.registries.jit_registry.register_jitted) decorator, 
which registers a Python function and the instructions on how to JIT compile it at the global registry 
[JITRegistry](/api/registries/jit_registry/#vectorbtpro.registries.jit_registry.JITRegistry).
The registry, once instructed, finds the function's setup and passes the function to a jitting class 
(aka "jitter") for wrapping. Preset jitters include [NumPyJitter](/api/utils/jitting/#vectorbtpro.utils.jitting.NumPyJitter) 
for NumPy implementations and [NumbaJitter](/api/utils/jitting/#vectorbtpro.utils.jitting.NumbaJitter) for 
Numba-compiled functions. The registry can also register tasks (by task id) and capture multiple jitter candidates 
for the same task. The user can then switch between different implementations by specifying `jitter`.

### Generic

- Refactored many methods that take UDFs (such as [GenericAccessor.rolling_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.rolling_apply)) 
by converting each into both a class (meta) and an instance (regular) method using 
[class_or_instancemethod](/api/utils/decorators/#vectorbtpro.utils.decorators.class_or_instancemethod).
If the method was called on an instance, its UDFs do not have to take any metadata apart from 
(a part of) the array, such as `apply_func_nb(window, *args)`. If the method was called on the class, 
it iterates over an abstract shape and its UDFs must take metadata of each iteration, which can be used 
to select a part of any custom array passed as a variable argument, such as `apply_func_nb(from_i, to_i, col, *args)`. 
Previously, UDFs had to accept both the metadata and the array, even if the metadata was not used.
- Most of the functions that take custom UDFs and variable arguments, such as 
[GenericAccessor.rolling_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.rolling_apply), 
received support for [templates](/api/utils/template/). The same goes for broadcasting named arguments - 
a practice initially introduced in [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio).
- Made crossovers more resilient to NaN and moved them to
[GenericAccessor.crossed_above](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_above)
and [GenericAccessor.crossed_below](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_below)
- Added [BaseAccessor.eval](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.eval), which is
similar to `pd.eval` but broadcasts inputs prior to evaluation and can switch between NumPy and NumExpr
- Improved conflict control in [MappedArray](/api/records/mapped_array/#vectorbtpro.records.mapped_array.MappedArray).
Multiple elements pointing to the same timestamp can be reduced using 
[Mapped.reduce_segments](/api/records/mapped_array/#vectorbtpro.records.mapped_array.MappedArray.reduce_segments).
They can also be safely converted to Pandas by repeating index.
- Made tolerance checks and values for Numba math functions such as 
[is_less_nb](/api/utils/math_/#vectorbtpro.utils.math_.is_less_nb) globally adjustable.
Disabling tolerance checks increases performance but can lead to round-off errors.
- Implemented context managers for profiling time ([Timer](/api/utils/profiling/#vectorbtpro.utils.profiling.Timer))
and memory ([MemTracer](/api/utils/profiling/#vectorbtpro.utils.profiling.MemTracer))
- Added support for [Scattergl](https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scattergl.html)
for plotting big datasets with increased speed. Used by default on more than 10,000 points.

### Broadcasting

- Refactored broadcasting mechanism inside of [broadcast](/api/base/reshaping/#vectorbtpro.base.reshaping.broadcast). 
Added [BCO](/api/base/reshaping/#vectorbtpro.base.reshaping.BCO) dataclass, whose instances can be 
passed to change broadcasting behavior for individual objects. Introduced a possibility to build a 
Cartesian product of scalar-like parameters and other broadcastable objects (both using `BCO` and 
`pd.Index` as a shortcut) using operation trees and 
[generate_param_combs](/api/utils/params/#vectorbtpro.utils.params.generate_param_combs).
Additionally, a random subset of parameter combinations can be automatically selected to emulate 
random search. [Default](/api/base/reshaping/#vectorbtpro.base.reshaping.Default) and 
[Ref](/api/base/reshaping/#vectorbtpro.base.reshaping.Ref) and the surrounding logic 
were moved to this module.

### Data

- Implemented data classes for working with local files: [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) 
for CSV files and [HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) for HDF files and keys. 
Both support efficient updates without having to read the entire file. To make this possible, 
symbol fetching methods can return a state, which are preserved for the use in data updates.
- Refactored [RandomData](/api/data/custom/#vectorbtpro.data.custom.RandomData) and 
[GBMData](/api/data/custom/#vectorbtpro.data.custom.GBMData)
- Better handling of missing data. Made [BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData) and 
[CCXTData](/api/data/custom/#vectorbtpro.data.custom.CCXTData) more error-resilient: in case of 
connectivity issues, data won't be lost but returned, so it can be updated later.
- Moved progress bar logic into a separate module to standardize handling of all progress bars 
across vectorbt. Added progress bar for symbols in [Data](/api/data/base/#vectorbtpro.data.base.Data).
- Renamed `download` to `fetch` everywhere since not all data sources reside online

### Portfolio

- Added a new simulation method 
[Portfolio.from_def_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_def_order_func)
that combines [Portfolio.from_orders](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_orders) 
and [Portfolio.from_order_func](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_order_func).
It uses a custom order function to transform array-like objects into orders and allows 
attaching and overriding user-defined callbacks to change and monitor simulation.
- Added support for in-output simulation objects. Instead of creating various arrays during the
simulation, they can be manually created by the user (or automatically created and broadcasted 
by utilizing templates) outside the simulation, passed as regular arguments, and modified in-place. 
They are then conveniently stored in [Portfolio.in_outputs](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.in_outputs) 
for further analysis. In addition, [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) 
can detect when an in-output array shadows a regular portfolio attribute and takes this array instead of 
reconstructing the attribute, which is the new way to efficiently precompute various artifacts such as returns.
- Implemented shortcut properties for [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio) 
and [Records](/api/records/base/#vectorbtpro.records.base.Records), which are cacheable properties that 
call their respective getter methods with default arguments. This enables dot notation such as
`pf.trades.winning.pnl.count()`, where `trades` and `winning` are cached properties that
call the [Portfolio.get_trades](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.get_trades) 
and [Trades.get_winning](/api/portfolio/trades/#vectorbtpro.portfolio.trades.Trades.get_winning) 
method respectively. In [Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio), 
shortcut properties can also utilize in-outputs.
- Made various portfolio attributes (such as [Portfolio.get_returns](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.get_returns)) 
flexible by converting each into both a class and an instance method using 
[class_or_instancemethod](/api/utils/decorators/#vectorbtpro.utils.decorators.class_or_instancemethod).
If the method was called on the class, the operation is run using the passed arguments only.
If the method was called on an instance, the operation is run on the data from the instance,
which can be overridden by setting any of the arguments.
- Introduced extra validation of arguments passed to simulation. For instance, passing arrays
that look boolean but have object data type raises an (informative) error.
- Not only [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
but all simulation functions accept `open`, `high`, and `low` (all optional).
This enables various interesting automatisms: order price of  `-np.inf` gets automatically 
replaced by the opening price and `np.inf` (default everywhere) by the closing price. 
The highest and lowest prices are being used for bar boundary checks.
- Added the following arguments:
    - `cash_deposits`: cash deposits/withdrawals at the beginning of each time step
    - `cash_earnings`: cash earnings (independent of position) at the end of each time step
    - `cash_dividends`: dividends (relative to position) at the end of each time step
    - `init_position`: the initial position
    - `signal_priority`: which signal to prioritize: stop or user?
- Allowed price of `0`. This allows for P&L-effective insertion and removal of cash and assets.
For instance, to periodically charge a fee, one can create a range of orders with zero price 
and non-zero fees. They are visible as regular trades and appear in records.
- Allowed `max_orders=0` and `max_logs=0` to disable filling records - for example, if the performance
is assessed during the simulation and there is no need to save this data for post-simulation analysis. 
Also, for parallelization reasons, both of the numbers refer to the maximal number of records **per 
column** rather than per entire input.
- Allowed negative fees (-0.05 means that you earn 0.05% per trade instead of paying a fee)
- Converted simulation outputs to named tuples of type [SimulationOutput](/api/portfolio/enums/#vectorbtpro.portfolio.enums.SimulationOutput)

### Returns

- Updated metrics based on returns to take into account datetime-like properties. For instance,
having two data points with the timestamps "2020-01-01" and "2021-01-01" are
considered as a full year rather than 2 days as it was previously. 
See [ArrayWrapper.dt_period](/api/base/wrapping/#vectorbtpro.base.wrapping.ArrayWrapper.dt_period).
- Rolling metrics such as [ReturnsAccessor.rolling_sortino_ratio](/api/returns/accessors/#vectorbtpro.returns.accessors.ReturnsAccessor.rolling_sortino_ratio)
were made much faster by refactoring their Numba-compiled functions

### Caching

- Completely refactored caching. Previously, caching was managed by specialized property and 
method decorators. Once the user invoked such a property or method, it checked for global settings to 
see whether it's blacklisted, and stored the cache on the instance it's bound to. Cached attributes
weren't easily discoverable, which led to less transparency. In the new approach, caching is being
managed by a global registry [CacheableRegistry](/api/registries/ca_registry/#vectorbtpro.registries.ca_registry.CacheableRegistry),
which registers so-called "setups" for all cacheable objects, such as functions, properties, methods, 
instances, and even classes. They all build a well-connected hierarchy that can propagate actions.
For instance, disabling caching in a class setup of type [CAClassSetup](/api/registries/ca_registry/#vectorbtpro.registries.ca_registry.CAClassSetup)
will disable caching across all of its child setups, down to [CARunSetup](/api/registries/ca_registry/#vectorbtpro.registries.ca_registry.CARunSetup),
which takes care of actual caching. Cacheable decorators such as [cacheable](/api/utils/decorators/#vectorbtpro.utils.decorators.cacheable) 
communicate with the registry and do all actions on their particular setup. The user can easily 
find the setup for any (cacheable) object to, for example, display various caching statistics.
- Removed caching of attributes that return DataFrames (apart from a few exceptions) to avoid wasting memory

### Design

- Restructured the project and reformatted the codebase. Most notably, Numba-compiled simulation 
functions were distributed across multiple modules.
- Some previously required packages such as Plotly and Dill were made optional to make
the core of vectorbt even more lightweight. Optional packages are tracked in 
[opt_packages](/api/utils/opt_packages/) and whenever a code that requires a package is accessed but 
the package is missing, an error is raised with instructions on how to install it.
- Converted minimalistic classes to dataclasses using [attrs](https://www.attrs.org/en/stable/)
- Refactored [Config](/api/utils/config/#vectorbtpro.utils.config.Config), which shrank 
initialization time of various vectorbt objects by 25%. Config respects Liskov substitution principle
and similar to a dict, can be initialized by using both positional and keyword arguments.
Also, created read-only and hybrid preset classes to unify configs created across vectorbt.
- Removed expected key checks, which makes subclassing vectorbt classes easier but
removes dynamic checks of keyword arguments passed to the initializer (which is an overkill anyway)
- Accessors were made cached by default (which can be changed in the settings) to avoid repeated
initialization, and all options for changing data in-place were removed
- Made [settings](/api/_settings/) more modular and better embeddable into documentation.
Additionally, upon import, vectorbt looks for an environment variable that contains the path 
to a settings file and updates/replaces the current settings in-place.
- Created and set up a private repository :sparkler:

*[SP]: Single-pass algorithm for best performance
*[NB]: Numba-compiled
