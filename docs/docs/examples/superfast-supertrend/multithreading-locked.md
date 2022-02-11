---
title: Multithreading
---

# :material-lock-open: Superfast SuperTrend - Multithreading

Having a purely Numba-compiled indicator function has one major benefit - multithreading support.
So, what exactly is multithreading and how it compares to multiprocessing?

Multithreading means having the same process run multiple threads **concurrently**, sharing the 
same CPU and memory. However, because of the [GIL](https://realpython.com/python-gil/) in Python, 
not all tasks can be executed faster by using multithreading. In fact, GIL allows only one thread 
to execute at a time even in a multi-threaded architecture with more than one CPU core, meaning 
only when one thread is idly waiting, another thread can start executing code.

To circumvent this limitation of the GIL, the most popular way is to use a multiprocessing approach 
where you use multiple processes instead of threads. Each Python process gets its own Python interpreter 
and memory space. And here's the catch: you cannot share the same array between two processes 
(you can, but it's tricky), and processes are (much) heavier than threads. For instance, vectorbt 
takes 2-3 seconds to be imported - are you willing to spend this much time in every single process? 
Such waiting time feels like eternity compared to our superfast streaming function.

But don't lose your faith just yet. Fortunately, compiled code called by the Python interpreter 
can release the GIL and execute on multiple threads at the same time. Libraries like NumPy and Pandas 
release the GIL automatically, while Numba requires the `nogil=True` flag to be set (as we luckily 
did above).

```pycon
>>> SuperTrend = vbt.IF(
...     class_name='SuperTrend',
...     short_name='st',
...     input_names=['high', 'low', 'close'],
...     param_names=['period', 'multiplier'],
...     output_names=['supert', 'superd', 'superl', 'supers']
... ).with_apply_func(
...     superfast_supertrend_nb, 
...     takes_1d=True,
...     jit_select_params=True,  # (1)!
...     jit_kwargs=dict(nogil=True),
...     period=7, 
...     multiplier=3
... )
```

1. Indicator factory dynamically generates a function that selects one parameter combination
at each time step and calls our `superfast_supertrend_nb`. These two arguments make it Numba-compiled
and release the GIL.

Let's benchmark this indicator on 336 parameter combinations per symbol:

```pycon
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(show_progress=False)
... )
269 ms ± 72.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

We see that each iteration takes around 270 / 336 / 2 = 400 microseconds, which is 2x slower than 
`superfast_supertrend_nb` itself. This is due to the fact that the indicator also has to concatenate 
all the generated columns of each output into a single array - apparently a costly operation.

Let's repeat the same test but now with multithreading enabled:

```pycon
>>> %%timeit
>>> SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True,
...     execute_kwargs=dict(
...         engine='dask',  # (1)!
...         chunk_len='auto',  # (2)!
...         show_progress=False  # (3)!
...     )
... )
147 ms ± 10.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

1. Use [Dask](https://dask.org/) as execution engine
2. Divide the entire parameter grid into chunks of the same length as the number
of cores in our system
3. Better not to show the progress bar when benchmarking

What the command did is the following: it divided all the parameter combinations into chunks.
Each chunk has the same number of combinations as we have cores, such that each of the combinations 
in that chunk can be executed concurrently. The chunks themselves are executed sequentially though. 
This way, we are always running at most `n` combinations and do not create more threads than needed. 

As we can see, this strategy has paid out with a 2x speedup.

[:material-lock: Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/locked-notebooks.md){ .md-button target="blank_" }
