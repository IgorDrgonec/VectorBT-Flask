# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Engines for executing functions."""

import multiprocessing
import concurrent.futures
import time

from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate, substitute_templates

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any

__all__ = [
    "SerialEngine",
    "ThreadPoolEngine",
    "ProcessPoolEngine",
    "PathosEngine",
    "DaskEngine",
    "RayEngine",
    "execute",
]


class ExecutionEngine(Configured):
    """Abstract class for executing functions."""

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        """Run an iterable of tuples out of a function, arguments, and keyword arguments.

        Provide `n_calls` in case `funcs_args` is a generator and the underlying engine needs it."""
        raise NotImplementedError


class SerialEngine(ExecutionEngine):
    """Class for executing functions sequentially.

    For defaults, see `engines.serial` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.serial"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "show_progress",
        "progress_desc",
        "pbar_kwargs",
        "clear_cache",
        "collect_garbage",
        "cooldown",
    }

    def __init__(
        self,
        progress_desc: tp.Optional[tp.Sequence] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Union[None, bool, int] = None,
        collect_garbage: tp.Union[None, bool, int] = None,
        cooldown: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        collect_garbage = self.resolve_setting(collect_garbage, "collect_garbage")
        cooldown = self.resolve_setting(cooldown, "cooldown")

        ExecutionEngine.__init__(
            self,
            show_progress=show_progress,
            progress_desc=progress_desc,
            pbar_kwargs=pbar_kwargs,
            clear_cache=clear_cache,
            collect_garbage=collect_garbage,
            cooldown=cooldown,
            **kwargs,
        )

        self._show_progress = show_progress
        self._progress_desc = progress_desc
        self._pbar_kwargs = pbar_kwargs
        self._clear_cache = clear_cache
        self._collect_garbage = collect_garbage
        self._cooldown = cooldown

    @property
    def progress_desc(self) -> tp.Optional[tp.Sequence]:
        """Sequence used to describe each iteration of the progress bar."""
        return self._progress_desc

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbtpro.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    @property
    def clear_cache(self) -> tp.Union[bool, int]:
        """Whether to clear vectorbt's cache after each iteration.

        If integer, do it once a number of calls."""
        return self._clear_cache

    @property
    def collect_garbage(self) -> tp.Union[bool, int]:
        """Whether to clear garbage after each iteration.

        If integer, do it once a number of calls."""
        return self._collect_garbage

    @property
    def cooldown(self) -> tp.Optional[int]:
        """Number of seconds to sleep after each call."""
        return self._cooldown

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.registries.ca_registry import clear_cache, collect_garbage

        results = []
        if n_calls is None and hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)
        with get_pbar(total=n_calls, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for i, (func, args, kwargs) in enumerate(funcs_args):
                if self.progress_desc is not None:
                    pbar.set_description(str(self.progress_desc[i]))
                results.append(func(*args, **kwargs))
                pbar.update(1)
                if isinstance(self.clear_cache, bool):
                    if self.clear_cache:
                        clear_cache()
                elif i > 0 and (i + 1) % self.clear_cache == 0:
                    clear_cache()
                if isinstance(self.collect_garbage, bool):
                    if self.collect_garbage:
                        collect_garbage()
                elif i > 0 and (i + 1) % self.collect_garbage == 0:
                    collect_garbage()
                if self.cooldown is not None:
                    time.sleep(self.cooldown)

        return results


class ThreadPoolEngine(ExecutionEngine):
    """Class for executing functions using `ThreadPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.threadpool` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.threadpool"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "timeout",
    }

    def __init__(self, init_kwargs: tp.KwargsLike = None, timeout: tp.Optional[int] = None, **kwargs) -> None:
        init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        timeout = self.resolve_setting(timeout, "timeout")

        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            **kwargs,
        )

        self._init_kwargs = init_kwargs
        self._timeout = timeout

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ThreadPoolExecutor`."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        with concurrent.futures.ThreadPoolExecutor(**self.init_kwargs) as executor:
            futures = {}
            for i, (func, args, kwargs) in enumerate(funcs_args):
                future = executor.submit(func, *args, **kwargs)
                futures[future] = i
            results = [None] * len(futures)
            for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                results[futures[fut]] = fut.result()
            return results


class ProcessPoolEngine(ExecutionEngine):
    """Class for executing functions using `ProcessPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.processpool` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.processpool"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "timeout",
    }

    def __init__(self, init_kwargs: tp.KwargsLike = None, timeout: tp.Optional[int] = None, **kwargs) -> None:
        init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        timeout = self.resolve_setting(timeout, "timeout")

        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            **kwargs,
        )

        self._init_kwargs = init_kwargs
        self._timeout = timeout

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ProcessPoolExecutor`."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        with concurrent.futures.ProcessPoolExecutor(**self.init_kwargs) as executor:
            futures = {}
            for i, (func, args, kwargs) in enumerate(funcs_args):
                future = executor.submit(func, *args, **kwargs)
                futures[future] = i
            results = [None] * len(futures)
            for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                results[futures[fut]] = fut.result()
            return results


def pass_kwargs_as_args(func, args, kwargs):
    """Helper function for `pathos.pools.ParallelPool`."""
    return func(*args, **kwargs)


class PathosEngine(ExecutionEngine):
    """Class for executing functions using `pathos`.

    For defaults, see `engines.pathos` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.pathos"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "pool_type",
        "init_kwargs",
        "timeout",
        "sleep",
        "show_progress",
        "pbar_kwargs",
        "join_pool",
    }

    def __init__(
        self,
        pool_type: tp.Optional[str] = None,
        init_kwargs: tp.KwargsLike = None,
        timeout: tp.Optional[int] = None,
        sleep: tp.Optional[float] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        join_pool: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        pool_type = self.resolve_setting(pool_type, "pool_type")
        init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        timeout = self.resolve_setting(timeout, "timeout")
        sleep = self.resolve_setting(sleep, "sleep")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        join_pool = self.resolve_setting(join_pool, "join_pool")

        ExecutionEngine.__init__(
            self,
            pool_type=pool_type,
            init_kwargs=init_kwargs,
            timeout=timeout,
            sleep=sleep,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            join_pool=join_pool,
            **kwargs,
        )

        self._pool_type = pool_type
        self._init_kwargs = init_kwargs
        self._timeout = timeout
        self._sleep = sleep
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._join_pool = join_pool

    @property
    def pool_type(self) -> str:
        """Pool type."""
        return self._pool_type

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize the pool."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    @property
    def sleep(self) -> tp.Optional[float]:
        """Number of seconds to sleep between checks."""
        return self._timeout

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbtpro.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    @property
    def join_pool(self) -> bool:
        """Whether to join the pool."""
        return self._join_pool

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pathos")

        if self.pool_type.lower() in ("thread", "threadpool"):
            from pathos.pools import ThreadPool as Pool
        elif self.pool_type.lower() in ("process", "processpool"):
            from pathos.pools import ProcessPool as Pool
        elif self.pool_type.lower() in ("parallel", "parallelpool"):
            from pathos.pools import ParallelPool as Pool

            funcs_args = [(pass_kwargs_as_args, x, {}) for x in funcs_args]
        else:
            raise ValueError(f"Invalid option pool_type='{self.pool_type}'")

        with Pool(**self.init_kwargs) as pool:
            async_results = []
            for func, args, kwargs in funcs_args:
                async_result = pool.apipe(func, *args, **kwargs)
                async_results.append(async_result)
            if self.timeout is not None or self.show_progress:
                pending = set(async_results)
                total_futures = len(pending)
                if self.timeout is not None:
                    end_time = self.timeout + time.monotonic()
                with get_pbar(total=total_futures, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
                    while pending:
                        pending = {async_result for async_result in pending if not async_result.ready()}
                        pbar.n = total_futures - len(pending)
                        pbar.refresh()
                        if len(pending) == 0:
                            break
                        if self.timeout is not None:
                            if time.monotonic() > end_time:
                                raise TimeoutError("%d (of %d) futures unfinished" % (len(pending), total_futures))
                        if self.sleep is not None:
                            time.sleep(self.sleep)
            if self.join_pool:
                pool.close()
                pool.join()
                pool.clear()
        return [async_result.get() for async_result in async_results]


class MpireEngine(ExecutionEngine):
    """Class for executing functions using `WorkerPool` from `mpire`.

    For defaults, see `engines.mpire` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.mpire"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "apply_kwargs",
        "timeout",
    }

    def __init__(
        self,
        init_kwargs: tp.KwargsLike = None,
        apply_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        apply_kwargs = self.resolve_setting(apply_kwargs, "apply_kwargs", merge=True)

        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            apply_kwargs=apply_kwargs,
            **kwargs,
        )

        self._init_kwargs = init_kwargs
        self._apply_kwargs = apply_kwargs

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `WorkerPool`."""
        return self._init_kwargs

    @property
    def apply_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `WorkerPool.async_apply`."""
        return self._apply_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("mpire")
        from mpire import WorkerPool

        with WorkerPool(**self.init_kwargs) as pool:
            async_results = []
            for i, (func, args, kwargs) in enumerate(funcs_args):
                async_result = pool.apply_async(func, args=args, kwargs=kwargs, **self.apply_kwargs)
                async_results.append(async_result)
            pool.stop_and_join()
        return [async_result.get() for async_result in async_results]


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    For defaults, see `engines.dask` in `vectorbtpro._settings.execution`.

    !!! note
        Use multi-threading mainly on numeric code that releases the GIL
        (like NumPy, Pandas, Scikit-Learn, Numba)."""

    _settings_path: tp.SettingsPath = "execution.engines.dask"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "compute_kwargs",
    }

    def __init__(self, compute_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        compute_kwargs = self.resolve_setting(compute_kwargs, "compute_kwargs", merge=True)

        ExecutionEngine.__init__(
            self,
            compute_kwargs=compute_kwargs,
            **kwargs,
        )

        self._compute_kwargs = compute_kwargs

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `dask.compute`."""
        return self._compute_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("dask")
        import dask

        results_delayed = []
        for func, args, kwargs in funcs_args:
            results_delayed.append(dask.delayed(func)(*args, **kwargs))
        return list(dask.compute(*results_delayed, **self.compute_kwargs))


class RayEngine(ExecutionEngine):
    """Class for executing functions in parallel using Ray.

    For defaults, see `engines.ray` in `vectorbtpro._settings.execution`.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup."""

    _settings_path: tp.SettingsPath = "execution.engines.ray"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "restart",
        "reuse_refs",
        "del_refs",
        "shutdown",
        "init_kwargs",
        "remote_kwargs",
    }

    def __init__(
        self,
        restart: tp.Optional[bool] = None,
        reuse_refs: tp.Optional[bool] = None,
        del_refs: tp.Optional[bool] = None,
        shutdown: tp.Optional[bool] = None,
        init_kwargs: tp.KwargsLike = None,
        remote_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        restart = self.resolve_setting(restart, "restart")
        reuse_refs = self.resolve_setting(reuse_refs, "reuse_refs")
        del_refs = self.resolve_setting(del_refs, "del_refs")
        shutdown = self.resolve_setting(shutdown, "shutdown")
        init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        remote_kwargs = self.resolve_setting(remote_kwargs, "remote_kwargs", merge=True)

        ExecutionEngine.__init__(
            self,
            restart=restart,
            reuse_refs=reuse_refs,
            del_refs=del_refs,
            shutdown=shutdown,
            init_kwargs=init_kwargs,
            remote_kwargs=remote_kwargs,
            **kwargs,
        )

        self._restart = restart
        self._reuse_refs = reuse_refs
        self._del_refs = del_refs
        self._shutdown = shutdown
        self._init_kwargs = init_kwargs
        self._remote_kwargs = remote_kwargs

    @property
    def restart(self) -> bool:
        """Whether to terminate the Ray runtime and initialize a new one."""
        return self._restart

    @property
    def reuse_refs(self) -> bool:
        """Whether to re-use function and object references, such that each unique object
        will be copied only once."""
        return self._reuse_refs

    @property
    def del_refs(self) -> bool:
        """Whether to explicitly delete the result object references."""
        return self._del_refs

    @property
    def shutdown(self) -> bool:
        """Whether to True to terminate the Ray runtime upon the job end."""
        return self._shutdown

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.init`."""
        return self._init_kwargs

    @property
    def remote_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.remote`."""
        return self._remote_kwargs

    @classmethod
    def get_ray_refs(
        cls,
        funcs_args: tp.FuncsArgs,
        reuse_refs: bool = True,
        remote_kwargs: tp.KwargsLike = None,
    ) -> tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Get result references by putting each argument and keyword argument into the object store
        and invoking the remote decorator on each function using Ray.

        If `reuse_refs` is True, will generate one reference per unique object id."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray
        from ray.remote_function import RemoteFunction
        from ray import ObjectRef

        if remote_kwargs is None:
            remote_kwargs = {}

        func_id_remotes = {}
        obj_id_refs = {}
        funcs_args_refs = []
        for func, args, kwargs in funcs_args:
            # Get remote function
            if isinstance(func, RemoteFunction):
                func_remote = func
            else:
                if not reuse_refs or id(func) not in func_id_remotes:
                    if isinstance(func, CPUDispatcher):
                        # Numba-wrapped function is not recognized by ray as a function
                        _func = lambda *_args, **_kwargs: func(*_args, **_kwargs)
                    else:
                        _func = func
                    if len(remote_kwargs) > 0:
                        func_remote = ray.remote(**remote_kwargs)(_func)
                    else:
                        func_remote = ray.remote(_func)
                    if reuse_refs:
                        func_id_remotes[id(func)] = func_remote
                else:
                    func_remote = func_id_remotes[id(func)]

            # Get id of each (unique) arg
            arg_refs = ()
            for arg in args:
                if isinstance(arg, ObjectRef):
                    arg_ref = arg
                else:
                    if not reuse_refs or id(arg) not in obj_id_refs:
                        arg_ref = ray.put(arg)
                        obj_id_refs[id(arg)] = arg_ref
                    else:
                        arg_ref = obj_id_refs[id(arg)]
                arg_refs += (arg_ref,)

            # Get id of each (unique) kwarg
            kwarg_refs = {}
            for kwarg_name, kwarg in kwargs.items():
                if isinstance(kwarg, ObjectRef):
                    kwarg_ref = kwarg
                else:
                    if not reuse_refs or id(kwarg) not in obj_id_refs:
                        kwarg_ref = ray.put(kwarg)
                        obj_id_refs[id(kwarg)] = kwarg_ref
                    else:
                        kwarg_ref = obj_id_refs[id(kwarg)]
                kwarg_refs[kwarg_name] = kwarg_ref

            funcs_args_refs.append((func_remote, arg_refs, kwarg_refs))
        return funcs_args_refs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray

        if self.restart:
            if ray.is_initialized():
                ray.shutdown()
        if not ray.is_initialized():
            ray.init(**self.init_kwargs)
        funcs_args_refs = self.get_ray_refs(funcs_args, reuse_refs=self.reuse_refs, remote_kwargs=self.remote_kwargs)
        result_refs = []
        for func_remote, arg_refs, kwarg_refs in funcs_args_refs:
            result_refs.append(func_remote.remote(*arg_refs, **kwarg_refs))
        try:
            results = ray.get(result_refs)
        finally:
            if self.del_refs:
                # clear object store
                del result_refs
            if self.shutdown:
                ray.shutdown()
        return results


def execute_serially(funcs_args: tp.FuncsArgs, id_objs: tp.Dict[int, tp.Any]) -> list:
    """Execute serially."""
    results = []
    for func, args, kwargs in funcs_args:
        new_func = id_objs[func]
        new_args = tuple(id_objs[arg] for arg in args)
        new_kwargs = {k: id_objs[v] for k, v in kwargs.items()}
        results.append(new_func(*new_args, **new_kwargs))
    return results


def build_serial_chunk(funcs_args: tp.FuncsArgs) -> tp.FuncArgs:
    """Build a serial chunk."""
    ref_ids = dict()
    id_objs = dict()

    def _prepare(x):
        if id(x) in ref_ids:
            return ref_ids[id(x)]
        new_id = len(id_objs)
        ref_ids[id(x)] = new_id
        id_objs[new_id] = x
        return new_id

    new_funcs_args = []
    for func, args, kwargs in funcs_args:
        new_func = _prepare(func)
        new_args = tuple(_prepare(arg) for arg in args)
        new_kwargs = {k: _prepare(v) for k, v in kwargs.items()}
        new_funcs_args.append((new_func, new_args, new_kwargs))
    return execute_serially, (new_funcs_args, id_objs), {}


def execute(
    funcs_args: tp.FuncsArgs,
    engine: tp.EngineLike = "serial",
    n_calls: tp.Optional[int] = None,
    n_chunks: tp.Optional[tp.Union[str, int]] = None,
    min_size: tp.Optional[int] = None,
    chunk_len: tp.Optional[tp.Union[str, int]] = None,
    chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    distribute: tp.Optional[str] = None,
    warmup: tp.Optional[bool] = None,
    in_chunk_order: bool = False,
    pre_execute_func: tp.Optional[tp.Callable] = None,
    pre_execute_kwargs: tp.KwargsLike = None,
    pre_chunk_func: tp.Optional[tp.Callable] = None,
    pre_chunk_kwargs: tp.KwargsLike = None,
    post_chunk_func: tp.Optional[tp.Callable] = None,
    post_chunk_kwargs: tp.KwargsLike = None,
    post_execute_func: tp.Optional[tp.Callable] = None,
    post_execute_kwargs: tp.KwargsLike = None,
    post_execute_on_sorted: bool = False,
    show_progress: tp.Optional[bool] = None,
    progress_desc: tp.Optional[tp.Sequence] = None,
    pbar_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
    engine_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> list:
    """Execute using an engine.

    Supported values for `engine`:

    * Name of the engine (see supported engines)
    * Subclass of `ExecutionEngine` - initializes with `kwargs` and `engine_kwargs`
    * Instance of `ExecutionEngine` - calls `ExecutionEngine.execute` with `n_calls`
    * Callable - passes `funcs_args`, `n_calls` (if not None), and `kwargs` and `engine_kwargs`

    Can execute per chunk if `chunk_meta` is provided. Otherwise, if any of `n_chunks` and `chunk_len`
    are set, passes them to `vectorbtpro.utils.chunking.Chunker.yield_chunk_meta` to generate `chunk_meta`.
    Arguments `n_chunks` and `chunk_len` can be set globally in the engine-specific settings.
    Set `n_chunks` and `chunk_len` to 'auto' to set them to the number of cores.

    If `distribute` is "calls", distributes calls within each chunk.
    If indices in `chunk_meta` are perfectly sorted and `funcs_args` is an iterable, iterates
    over `funcs_args` to avoid converting it into a list. Otherwise, iterates over `chunk_meta`.
    If `in_chunk_order` is True, returns the outputs in the order they appear in `chunk_meta`.
    Otherwise, always returns them in the same order as in `funcs_args`.

    If `distribute` is "chunks", distributes chunks. For this, executes calls
    within each chunk serially using `execute_serially`. Also, compresses each chunk such that
    each unique function, positional argument, and keyword argument is serialized only once.

    If `funcs_args` is a custom template, substitutes it once `chunk_meta` is established.
    Use `template_context` as an additional context. All the resolved functions and arguments
    will be immediately passed to the executor.

    If `pre_chunk_func` is not None, calls the function before processing a chunk. If it returns anything
    other than None, the returned object will be appended to the outputs and the chunk won't be executed.
    This enables use cases such as caching. If `post_chunk_func` is not None, calls the function after
    processing the chunk. It should return either None to keep the old call outputs, or return new ones.
    Will also substitute any templates in `pre_chunk_kwargs` and `post_chunk_kwargs` and pass them as
    keyword arguments. The following additional arguments are available in the contexts: the index of
    the current chunk `chunk_idx`, the list of call indices `call_indices` in the chunk, the list of
    call outputs `call_outputs` returned by executing the chunk (only for `post_chunk_func`), and
    whether the chunk was executed `chunk_executed` or otherwise returned by `pre_chunk_func`
    (only for `post_chunk_func`).

    !!! note
        The both callbacks above are effective only when `distribute` is "calls" and chunking is enabled.

    If `pre_execute_func` is not None, calls the function before processing all calls. Should return
    nothing (None). Will also substitute any templates in `post_execute_kwargs` and pass them as keyword
    arguments. The following additional arguments are available in the context: the number of chunks `n_chunks`.

    If `post_execute_func` is not None, calls the function after processing all calls. Will also substitute
    any templates in `post_execute_kwargs` and pass them as keyword arguments. Should return either None
    to keep the default outputs or return the new ones. The following additional arguments are available
    in the context: the number of chunks `n_chunks` and the generated flattened list of outputs `outputs`.
    If `post_execute_on_sorted` is True, will run the callback after sorting the call indices.

    !!! info
        Chunks are processed sequentially, while functions within each chunk can be processed distributively.

    Supported engines can be found in `engines` in `vectorbtpro._settings.execution`."""
    from vectorbtpro._settings import settings

    execution_cfg = settings["execution"]
    engines_cfg = execution_cfg["engines"]

    engine_kwargs = merge_dicts(kwargs, engine_kwargs)

    engine_cfg = dict()
    if isinstance(engine, str):
        if engine.lower() in engines_cfg:
            engine_cfg = engines_cfg[engine]
            engine = engines_cfg[engine]["cls"]
        else:
            raise ValueError(f"Invalid engine name '{engine}'")
    if isinstance(engine, str):
        globals_dict = globals()
        if engine in globals_dict:
            engine = globals_dict[engine]
        else:
            raise ValueError(f"Invalid engine name '{engine}'")
    if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is engine:
                engine_cfg = v
        func_arg_names = get_func_arg_names(engine.__init__)
        if "show_progress" in func_arg_names and "show_progress" not in engine_kwargs:
            engine_kwargs["show_progress"] = show_progress
        if "progress_desc" in func_arg_names and "progress_desc" not in engine_kwargs:
            engine_kwargs["progress_desc"] = progress_desc
        if "pbar_kwargs" in func_arg_names and "pbar_kwargs" not in engine_kwargs:
            engine_kwargs["pbar_kwargs"] = pbar_kwargs
        engine = engine(**engine_kwargs)
    elif isinstance(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is type(engine):
                engine_cfg = v
    if callable(engine):
        func_arg_names = get_func_arg_names(engine)
        if "show_progress" in func_arg_names and "show_progress" not in engine_kwargs:
            engine_kwargs["show_progress"] = show_progress
        if "progress_desc" in func_arg_names and "progress_desc" not in engine_kwargs:
            engine_kwargs["progress_desc"] = progress_desc
        if "pbar_kwargs" in func_arg_names and "pbar_kwargs" not in engine_kwargs:
            engine_kwargs["pbar_kwargs"] = pbar_kwargs

    if n_chunks is None:
        n_chunks = engine_cfg.get("n_chunks", execution_cfg["n_chunks"])
    if min_size is None:
        min_size = engine_cfg.get("min_size", execution_cfg["min_size"])
    if chunk_len is None:
        chunk_len = engine_cfg.get("chunk_len", execution_cfg["chunk_len"])
    if distribute is None:
        distribute = engine_cfg.get("distribute", execution_cfg["distribute"])
    if warmup is None:
        warmup = engine_cfg.get("warmup", execution_cfg["warmup"])
    if warmup:
        if not hasattr(funcs_args, "__getitem__"):
            funcs_args = list(funcs_args)
        funcs_args[0][0](*funcs_args[0][1], **funcs_args[0][2])
    if show_progress is None:
        show_progress = engine_cfg.get("show_progress", execution_cfg["show_progress"])
    if pre_execute_func is None:
        pre_execute_func = engine_cfg.get("pre_execute_func", execution_cfg["pre_execute_func"])
    pre_execute_kwargs = merge_dicts(
        execution_cfg["pre_execute_kwargs"],
        engine_cfg.get("pre_execute_kwargs", None),
        pre_execute_kwargs,
    )
    if pre_chunk_func is None:
        pre_chunk_func = engine_cfg.get("pre_chunk_func", execution_cfg["pre_chunk_func"])
    pre_chunk_kwargs = merge_dicts(
        execution_cfg["pre_chunk_kwargs"],
        engine_cfg.get("pre_chunk_kwargs", None),
        pre_chunk_kwargs,
    )
    if post_chunk_func is None:
        post_chunk_func = engine_cfg.get("post_chunk_func", execution_cfg["post_chunk_func"])
    post_chunk_kwargs = merge_dicts(
        execution_cfg["post_chunk_kwargs"],
        engine_cfg.get("post_chunk_kwargs", None),
        post_chunk_kwargs,
    )
    if post_execute_func is None:
        post_execute_func = engine_cfg.get("post_execute_func", execution_cfg["post_execute_func"])
    post_execute_kwargs = merge_dicts(
        execution_cfg["post_execute_kwargs"],
        engine_cfg.get("post_execute_kwargs", None),
        post_execute_kwargs,
    )
    if post_execute_on_sorted is None:
        post_execute_on_sorted = engine_cfg.get("post_execute_on_sorted", execution_cfg["post_execute_on_sorted"])
    pbar_kwargs = merge_dicts(
        execution_cfg["pbar_kwargs"],
        engine_cfg.get("pbar_kwargs", None),
        pbar_kwargs,
    )
    template_context = merge_dicts(
        execution_cfg["template_context"],
        engine_cfg.get("template_context", None),
        template_context,
    )

    def _call_pre_execute_func():
        if pre_execute_func is not None:
            _pre_execute_func = substitute_templates(
                pre_execute_func,
                template_context,
                sub_id="pre_execute_func",
            )
            _pre_execute_kwargs = substitute_templates(
                pre_execute_kwargs,
                template_context,
                sub_id="pre_execute_kwargs",
            )
            _pre_execute_func(**_pre_execute_kwargs)

    def _execute(funcs_args, n_calls):
        if isinstance(engine, ExecutionEngine):
            return engine.execute(funcs_args, n_calls=n_calls)
        if callable(engine):
            if "n_calls" in func_arg_names:
                return engine(funcs_args, n_calls=n_calls, **engine_kwargs)
            return engine(funcs_args, **engine_kwargs)
        raise TypeError(f"Engine of type {type(engine)} is not supported")

    def _call_post_execute_func(outputs):
        if post_execute_func is not None:
            _template_context = merge_dicts(
                dict(
                    outputs=outputs,
                ),
                template_context,
            )
            _post_execute_func = substitute_templates(
                post_execute_func,
                _template_context,
                sub_id="post_execute_func",
            )
            _post_execute_kwargs = substitute_templates(
                post_execute_kwargs,
                _template_context,
                sub_id="post_execute_kwargs",
            )
            new_outputs = _post_execute_func(**_post_execute_kwargs)
            if new_outputs is not None:
                return new_outputs
        return outputs

    if n_chunks is None and chunk_len is None and chunk_meta is None:
        n_chunks = 1
    if n_chunks == 1 and not isinstance(funcs_args, CustomTemplate):
        _call_pre_execute_func()
        if "n_chunks" not in template_context:
            template_context["n_chunks"] = 1
        return _call_post_execute_func(_execute(funcs_args, n_calls))

    if chunk_meta is None:
        # Generate chunk metadata
        from vectorbtpro.utils.chunking import Chunker

        if not isinstance(funcs_args, CustomTemplate) and hasattr(funcs_args, "__len__"):
            _n_calls = len(funcs_args)
        elif n_calls is not None:
            _n_calls = n_calls
        else:
            if isinstance(funcs_args, CustomTemplate):
                raise ValueError("When funcs_args is a template, must provide n_calls")
            funcs_args = list(funcs_args)
            _n_calls = len(funcs_args)
        if isinstance(n_chunks, str) and n_chunks.lower() == "auto":
            n_chunks = multiprocessing.cpu_count()
        if isinstance(chunk_len, str) and chunk_len.lower() == "auto":
            chunk_len = multiprocessing.cpu_count()
        chunk_meta = Chunker.yield_chunk_meta(
            n_chunks=n_chunks,
            size=_n_calls,
            min_size=min_size,
            chunk_len=chunk_len,
        )
        if "chunk_meta" not in template_context:
            template_context["chunk_meta"] = chunk_meta

    # Substitute templates
    if isinstance(funcs_args, CustomTemplate):
        funcs_args = substitute_templates(funcs_args, template_context, sub_id="funcs_args")
        if hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)
        else:
            n_calls = None
        _call_pre_execute_func()
        if "n_chunks" not in template_context:
            template_context["n_chunks"] = 1
        return _call_post_execute_func(_execute(funcs_args, n_calls))

    # Get indices of each chunk and whether they are sorted
    last_idx = -1
    indices_sorted = True
    all_call_indices = []
    for _chunk_meta in chunk_meta:
        if _chunk_meta.indices is not None:
            call_indices = list(_chunk_meta.indices)
        else:
            if _chunk_meta.start is None or _chunk_meta.end is None:
                raise ValueError("Each chunk must have a start and an end index")
            call_indices = list(range(_chunk_meta.start, _chunk_meta.end))
        if indices_sorted:
            for idx in call_indices:
                if idx != last_idx + 1:
                    indices_sorted = False
                    break
                last_idx = idx
        all_call_indices.append(call_indices)
    if "n_chunks" not in template_context:
        template_context["n_chunks"] = len(all_call_indices)

    if distribute.lower() == "calls":

        def _call_pre_chunk_func(chunk_idx, call_indices):
            if pre_chunk_func is not None:
                _template_context = merge_dicts(
                    dict(
                        chunk_idx=chunk_idx,
                        call_indices=call_indices,
                    ),
                    template_context,
                )
                _pre_chunk_func = substitute_templates(
                    pre_chunk_func,
                    _template_context,
                    sub_id="pre_chunk_func",
                )
                _pre_chunk_kwargs = substitute_templates(
                    pre_chunk_kwargs,
                    _template_context,
                    sub_id="pre_chunk_kwargs",
                )
                return _pre_chunk_func(**_pre_chunk_kwargs)
            return None

        def _call_post_chunk_func(chunk_idx, call_indices, call_outputs, chunk_executed):
            if post_chunk_func is not None:
                _template_context = merge_dicts(
                    dict(
                        chunk_idx=chunk_idx,
                        call_indices=call_indices,
                        call_outputs=call_outputs,
                        chunk_executed=chunk_executed,
                    ),
                    template_context,
                )
                _post_chunk_func = substitute_templates(
                    post_chunk_func,
                    _template_context,
                    sub_id="post_chunk_func",
                )
                _post_chunk_kwargs = substitute_templates(
                    post_chunk_kwargs,
                    _template_context,
                    sub_id="post_chunk_kwargs",
                )
                new_call_outputs = _post_chunk_func(**_post_chunk_kwargs)
                if new_call_outputs is not None:
                    return new_call_outputs
            return call_outputs

        if indices_sorted and not hasattr(funcs_args, "__len__"):
            # Iterate over funcs_args
            outputs = []
            chunk_idx = 0
            _funcs_args = []

            _call_pre_execute_func()
            with get_pbar(total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                for i, func_args in enumerate(funcs_args):
                    if i > all_call_indices[chunk_idx][-1]:
                        call_indices = all_call_indices[chunk_idx]
                        call_outputs = _call_pre_chunk_func(chunk_idx, call_indices)
                        if call_outputs is None:
                            call_outputs = _execute(_funcs_args, len(call_indices))
                            chunk_executed = True
                        else:
                            chunk_executed = False
                        call_outputs = _call_post_chunk_func(chunk_idx, call_indices, call_outputs, chunk_executed)
                        outputs.extend(call_outputs)
                        chunk_idx += 1
                        _funcs_args = []
                        pbar.update(1)
                    _funcs_args.append(func_args)
                if len(_funcs_args) > 0:
                    call_indices = all_call_indices[chunk_idx]
                    call_outputs = _call_pre_chunk_func(chunk_idx, call_indices)
                    if call_outputs is None:
                        call_outputs = _execute(_funcs_args, len(call_indices))
                        chunk_executed = True
                    else:
                        chunk_executed = False
                    call_outputs = _call_post_chunk_func(chunk_idx, call_indices, call_outputs, chunk_executed)
                    outputs.extend(call_outputs)
                    pbar.update(1)
            return _call_post_execute_func(outputs)
        else:
            # Iterate over chunks
            funcs_args = list(funcs_args)
            outputs = []
            output_indices = []

            _call_pre_execute_func()
            with get_pbar(total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                for chunk_idx, call_indices in enumerate(all_call_indices):
                    call_outputs = _call_pre_chunk_func(chunk_idx, call_indices)
                    if call_outputs is None:
                        _funcs_args = []
                        for idx in call_indices:
                            _funcs_args.append(funcs_args[idx])
                        call_outputs = _execute(_funcs_args, len(call_indices))
                        chunk_executed = True
                    else:
                        chunk_executed = False
                    call_outputs = _call_post_chunk_func(chunk_idx, call_indices, call_outputs, chunk_executed)
                    outputs.extend(call_outputs)
                    output_indices.extend(call_indices)
                    pbar.update(1)
            if not post_execute_on_sorted:
                outputs = _call_post_execute_func(outputs)
            if not in_chunk_order and not indices_sorted:
                outputs = [x for _, x in sorted(zip(output_indices, outputs))]
            if post_execute_on_sorted:
                outputs = _call_post_execute_func(outputs)
            return outputs

    elif distribute.lower() == "chunks":
        if indices_sorted and not hasattr(funcs_args, "__len__"):
            # Iterate over funcs_args
            chunk_idx = 0
            _funcs_args = []
            funcs_args_chunks = []

            _call_pre_execute_func()
            for i, func_args in enumerate(funcs_args):
                if i > all_call_indices[chunk_idx][-1]:
                    funcs_args_chunks.append(build_serial_chunk(_funcs_args))
                    chunk_idx += 1
                    _funcs_args = []
                _funcs_args.append(func_args)
            if len(_funcs_args) > 0:
                funcs_args_chunks.append(build_serial_chunk(_funcs_args))
            outputs = _execute(funcs_args_chunks, len(funcs_args_chunks))
            outputs = [x for o in outputs for x in o]
            return _call_post_execute_func(outputs)
        else:
            # Iterate over chunks
            funcs_args = list(funcs_args)
            funcs_args_chunks = []
            output_indices = []

            _call_pre_execute_func()
            for call_indices in all_call_indices:
                _funcs_args = []
                for idx in call_indices:
                    _funcs_args.append(funcs_args[idx])
                funcs_args_chunks.append(build_serial_chunk(_funcs_args))
                output_indices.extend(call_indices)
            outputs = _execute(funcs_args_chunks, len(funcs_args_chunks))
            outputs = [x for o in outputs for x in o]
            if not post_execute_on_sorted:
                outputs = _call_post_execute_func(outputs)
            if not in_chunk_order and not indices_sorted:
                outputs = [x for _, x in sorted(zip(output_indices, outputs))]
            if post_execute_on_sorted:
                outputs = _call_post_execute_func(outputs)
            return outputs
    else:
        raise ValueError(f"Invalid option distribute='{distribute}'")
