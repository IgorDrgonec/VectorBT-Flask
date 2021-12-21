---
title: Features
---

# Features :zap:

On top of the [features](https://github.com/polakowo/vectorbt#zap-features) offered by the vanilla version of vectorbt, vectorbt PRO implements the following enhancements:

## Performance

- [x] **Parallelization with Numba**: Most Numba-compiled functions have been rewritten to process columns in parallel using [Explicit Parallel Loops](https://numba.pydata.org/numba-doc/0.37.0/user/parallel.html#explicit-parallel-loops), which can be enabled by a single command.
- [x] **Chunking**: Innovative chunking mechanism that takes a specification of how arguments should be chunked, automatically splits them, passes each chunk to the function, and merges back the results. This way, you can run any function in a distributed manner! Additionally, vectorbt PRO provides the chunking specification for all arguments of most Numba-compiled functions. Chunking can be enabled by a single command. No more out-of-memory errors!
- [x] **Multithreading**: Integration of the [Dask](https://dask.org/) backend for running multiple chunks across multiple threads. Cuts down execution time on Apple M1 by 3-4x, and even more depending on the number of cores. Dask + Numba = :muscle:
- [x] **Multiprocessing**: Integration of the [Ray](https://www.ray.io/) backend for running multiple chunks across multiple processes. Ever wanted to test billions of hyperparameter combinations in a matter of minutes? This is now possible by scaling functions and entire applications up in the cloud using the Ray cluster. :eyes:
- [x] **Jitting**: Jitting means just-in-time compiling. Although Numba remains the primary jitter in vectorbt, vectorbt PRO now enables implementation of custom jitter classes such as that for [JAX](https://github.com/google/jax) with GPU support. Every jitted function is registered globally so you can switch between implementations using a single command.
- [x] **Caching**: Caching has been reimplemented from the ground up and now it's being managed by a single registry. This allows for tracking useful statistics of all cacheable parts of vectorbt, such as to display the total cached size in MB. Full control and transparency.

## Flexibility

- [x] **Smart broadcasting**: Broadcasting mechanism has been completely refactored and now supports parameters. Build a product of multiple hyperparameter combinations with a single line of code. :magic_wand:
- [x] **Meta methods**: Many methods such as rolling apply are now available in two flavors: regular (instance methods) and meta (class methods). Regular methods work as usual, while meta methods are not bound to any array and act as micro-pipelines with their own broadcasting and templating logic.

## Data

- [x] **Refactored data**: In case of connectivity issues, data won't be lost but returned so it can be updated later. Also, symbol fetching methods can also return a state, which will be preserved for the use in data updates. There is also a new progress bar for symbols.
- [x] **Local data**: Added data classes that specialize in loading data from files, such as HDF5 or CSV. 

## Modeling

- [x] **More checks**: Introduced typing and OHLC violation checks.
- [x] **Initial position**: Similar to initial cash, initial position can now be specified.
- [x] **Cash deposits**: Cash can now be deposited/withdrawn at any time. 
- [x] **Cash earnings**: Cash earnings and dividends can now be added/removed at any time.
- [x] **Shortcut properties**: Portfolio simulation based on order functions can take user-defined arrays, write them in place during the simulation, and use them directly instead of calculating them during the analysis phase. This is the new, very convinient way to pre-calculate and utilize attributes such as returns.

## Analysis

- [x] **Hyperfast rolling metrics**: Rolling metrics based on returns have been optimized for best performance (up to 100x speedup).
- [x] **Flexible portfolio attributes**: Portfolio attributes such as asset flow now allow overriding other attributes they are based on. This allows great control of post-simulation analysis, for example, to eliminate re-calculation.
- [x] **Redesigned records**: Every records class now has a config that describes each field of the NumPy records array. This enables many useful automatisms, such as type checks and construction of human-readable DataFrames. This also makes it far easier to subclass other records classes.

## Settings

- [x] **Optimized config**: Config classes have been optimized. The time overhead of using the top-level API has shrunk by 25%.
- [x] **New settings**: Settings have become more intuitive and human-readable. You can also use environment variables to instruct vectorbt to load settings from a file before any function is registered internally. This has some interesting use cases, such as overriding Numba options for specific functions and even replacing entire implementations in the deepest layers of the vectorbt's core.

