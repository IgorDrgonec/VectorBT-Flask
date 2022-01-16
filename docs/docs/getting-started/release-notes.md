---
title: Release notes
---

# Release notes

All notable changes in reverse chronological order.

## Version 1.0.7 (16 January, 2021)

- Changed `np.int_` to `np.integer` when passed to `np.issubdtype`
- Refactored auto-aligned initial cash to be based on free cash flows and cash deposits
- Upgraded the string parser of [deep_getattr](/api/utils/attr_/#vectorbtpro.utils.attr_.deep_getattr) 
to accept strings of chained method calls (more intuitive)
- Implemented superfast Pearson correlation coefficient and its rolling version
- Created the class [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable),
which combines [Wrapping](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping) and builder mixins
- Metrics and subplots that require a single column won't raise an error if the object
is two-dimensional and has only one column
- [Grouper](/api/base/grouping/#vectorbtpro.base.grouping.Grouper) can return a group map,
which isn't tied to a strict group ordering and is easier to use outside of Numba

## Version 1.0.6 (9 January, 2021)

- Benchmark can be disabled by passing `bm_close=False` to 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio)

## Version 1.0.5 (8 January, 2021)

- Fixed [Portfolio.from_signals](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals) 
for `direction='both'` and `size_type='value'`. Previously, the position couldn't be properly reversed.
- Avoid sorting paths in [LocalData](/api/data/custom/#vectorbtpro.data.custom.LocalData) 
if they are passed as a sequence

## Version 1.0.4 (6 January, 2021)

- Set benchmark easily and also globally (#32) by passing `bm_close` to 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio).
Works similarly to [Portfolio.close](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.close).
- Shortened registry names (such as from `ca_registry` to `ca_reg`)

## Version 1.0.3 (5 January, 2021)

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

## Version 1.0.2 (31 December, 2021)

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

## Version 1.0.1 (21 December, 2021)

- Adapted the codebase to the new documentation format
- Upgraded the documentation website generator from pdoc3 to MkDocs (Material Insiders). 
API is being automatically converted to Markdown files by a modified version of pdoc3 that 
resides in a private repository of @polakowo.

## Version 1.0.0 (13 December, 2021)

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
