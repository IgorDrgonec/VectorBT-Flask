# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Engines for executing functions."""

import multiprocessing

from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.parsing import get_func_arg_names

try:
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any


class ExecutionEngine(Configured):
    """Abstract class for executing functions."""

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        """Run an iterable of tuples out of a function, arguments, and keyword arguments.

        Provide `n_calls` in case `funcs_args` is a generator and the underlying engine needs it."""
        raise NotImplementedError


class SequenceEngine(ExecutionEngine):
    """Class for executing functions sequentially.

    For defaults, see `engines.sequence` in `vectorbtpro._settings.execution`."""

    def __init__(self, show_progress: tp.Optional[bool] = None, pbar_kwargs: tp.KwargsLike = None) -> None:
        from vectorbtpro._settings import settings

        sequence_cfg = settings["execution"]["engines"]["sequence"]

        if show_progress is None:
            show_progress = sequence_cfg["show_progress"]
        pbar_kwargs = merge_dicts(pbar_kwargs, sequence_cfg["pbar_kwargs"])

        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

        ExecutionEngine.__init__(self, show_progress=show_progress, pbar_kwargs=pbar_kwargs)

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbtpro.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        results = []
        if n_calls is None and hasattr(funcs_args, "__len__"):
            n_calls = len(funcs_args)
        with get_pbar(total=n_calls, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for func, args, kwargs in funcs_args:
                results.append(func(*args, **kwargs))
                pbar.update(1)
        return results


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    For defaults, see `engines.dask` in `vectorbtpro._settings.execution`.

    !!! note
        Use multi-threading mainly on numeric code that releases the GIL
        (like NumPy, Pandas, Scikit-Learn, Numba)."""

    def __init__(self, **compute_kwargs) -> None:
        from vectorbtpro._settings import settings

        dask_cfg = settings["execution"]["engines"]["dask"]

        compute_kwargs = merge_dicts(compute_kwargs, dask_cfg["compute_kwargs"])

        self._compute_kwargs = compute_kwargs

        ExecutionEngine.__init__(self, **compute_kwargs)

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `dask.compute`."""
        return self._compute_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbtpro.utils.opt_packages import assert_can_import

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

    def __init__(
        self,
        restart: tp.Optional[bool] = None,
        reuse_refs: tp.Optional[bool] = None,
        del_refs: tp.Optional[bool] = None,
        shutdown: tp.Optional[bool] = None,
        init_kwargs: tp.KwargsLike = None,
        remote_kwargs: tp.KwargsLike = None,
    ) -> None:
        from vectorbtpro._settings import settings

        ray_cfg = settings["execution"]["engines"]["ray"]

        if restart is None:
            restart = ray_cfg["restart"]
        if reuse_refs is None:
            reuse_refs = ray_cfg["reuse_refs"]
        if del_refs is None:
            del_refs = ray_cfg["del_refs"]
        if shutdown is None:
            shutdown = ray_cfg["shutdown"]
        init_kwargs = merge_dicts(init_kwargs, ray_cfg["init_kwargs"])
        remote_kwargs = merge_dicts(remote_kwargs, ray_cfg["remote_kwargs"])

        self._restart = restart
        self._reuse_refs = reuse_refs
        self._del_refs = del_refs
        self._shutdown = shutdown
        self._init_kwargs = init_kwargs
        self._remote_kwargs = remote_kwargs

        ExecutionEngine.__init__(
            self,
            restart=restart,
            reuse_refs=reuse_refs,
            del_refs=del_refs,
            shutdown=shutdown,
            init_kwargs=init_kwargs,
            remote_kwargs=remote_kwargs,
        )

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

    @staticmethod
    def get_ray_refs(
        funcs_args: tp.FuncsArgs,
        reuse_refs: bool = True,
        remote_kwargs: tp.KwargsLike = None,
    ) -> tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Get result references by putting each argument and keyword argument into the object store
        and invoking the remote decorator on each function using Ray.

        If `reuse_refs` is True, will generate one reference per unique object id."""
        from vectorbtpro.utils.opt_packages import assert_can_import

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
        from vectorbtpro.utils.opt_packages import assert_can_import

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


def execute(
    funcs_args: tp.FuncsArgs,
    engine: tp.EngineLike = SequenceEngine,
    n_calls: tp.Optional[int] = None,
    n_chunks: tp.Optional[int] = None,
    chunk_len: tp.Optional[tp.Union[str, int]] = None,
    chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    in_chunk_order: bool = False,
    show_progress: tp.Optional[bool] = None,
    pbar_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> list:
    """Execute using an engine.

    Supported values for `engine`:

    * Name of the engine (see supported engines)
    * Subclass of `ExecutionEngine` - will initialize with `**kwargs`
    * Instance of `ExecutionEngine` - will call `ExecutionEngine.execute` with `n_calls`
    * Callable - will pass `funcs_args`, `n_calls` (if not None), and `**kwargs`

    Can execute per chunk if `chunk_meta` is provided. Otherwise, if any of `n_chunks` and `chunk_len`
    are set, passes them to `vectorbtpro.utils.chunking.yield_chunk_meta` to generate `chunk_meta`.
    Arguments `n_chunks` and `chunk_len` can be set globally in the engine-specific settings.
    Set `chunk_len` to 'auto' to set the chunk length to the number of cores.

    If indices in `chunk_meta` are perfectly sorted and `funcs_args` is an iterable, traverses
    through `funcs_args` to avoid converting it into a list. Otherwise, traverses through `chunk_meta`.
    If `in_chunk_order` is True, returns the outputs in the order they appear in `chunk_meta`.
    Otherwise, always returns them in the same order as in `funcs_args`.

    !!! info
        Chunks are processed sequentially, while functions within each chunk can be processed distributively.

    Supported engines can be found in `engines` in `vectorbtpro._settings.execution`."""
    from vectorbtpro._settings import settings

    execution_cfg = settings["execution"]
    engines_cfg = execution_cfg["engines"]

    engine_cfg = dict()
    if isinstance(engine, str):
        if engine.lower() in engines_cfg:
            engine_cfg = engines_cfg[engine]
            engine = engines_cfg[engine]["cls"]
        else:
            raise ValueError(f"Engine with name '{engine}' is unknown")
    if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is engine:
                engine_cfg = v
        func_arg_names = get_func_arg_names(engine.__init__)
        if "show_progress" in func_arg_names:
            kwargs["show_progress"] = show_progress
        if "pbar_kwargs" in func_arg_names:
            kwargs["pbar_kwargs"] = pbar_kwargs
        engine = engine(**kwargs)
    elif isinstance(engine, ExecutionEngine):
        for k, v in engines_cfg.items():
            if v["cls"] is type(engine):
                engine_cfg = v
    if callable(engine):
        func_arg_names = get_func_arg_names(engine)
        if "show_progress" in func_arg_names:
            kwargs["show_progress"] = show_progress
        if "pbar_kwargs" in func_arg_names:
            kwargs["pbar_kwargs"] = pbar_kwargs

    if n_chunks is None:
        n_chunks = engine_cfg.get("n_chunks", None)
    if chunk_len is None:
        chunk_len = engine_cfg.get("chunk_len", None)
    if show_progress is None:
        show_progress = execution_cfg["show_progress"]
    pbar_kwargs = merge_dicts(execution_cfg["pbar_kwargs"], pbar_kwargs)

    def _execute(_funcs_args: tp.FuncsArgs, _n_calls: tp.Optional[int]) -> list:
        if isinstance(engine, ExecutionEngine):
            return engine.execute(_funcs_args, n_calls=_n_calls)
        if callable(engine):
            if n_calls is not None:  # use global n_calls
                return engine(_funcs_args, n_calls=_n_calls, **kwargs)
            return engine(_funcs_args, **kwargs)
        raise TypeError(f"Engine of type {type(engine)} is not supported")

    if n_chunks is None and chunk_len is None and chunk_meta is None:
        return _execute(funcs_args, n_calls)

    if chunk_meta is None:
        # Generate chunk metadata
        from vectorbtpro.utils.chunking import yield_chunk_meta

        if hasattr(funcs_args, "__len__"):
            _n_calls = len(funcs_args)
        elif n_calls is not None:
            _n_calls = n_calls
        else:
            funcs_args = list(funcs_args)
            _n_calls = len(funcs_args)
        if isinstance(chunk_len, str) and chunk_len.lower() == "auto":
            chunk_len = multiprocessing.cpu_count()
        chunk_meta = yield_chunk_meta(n_chunks=n_chunks, size=_n_calls, chunk_len=chunk_len)

    # Get indices of each chunk and whether they are sorted
    last_idx = -1
    indices_sorted = True
    all_chunk_indices = []
    for _chunk_meta in chunk_meta:
        if _chunk_meta.indices is not None:
            chunk_indices = list(_chunk_meta.indices)
        else:
            if _chunk_meta.start is None or _chunk_meta.end is None:
                raise ValueError("Each chunk must have a start and an end index")
            chunk_indices = list(range(_chunk_meta.start, _chunk_meta.end))
        if indices_sorted:
            for idx in chunk_indices:
                if idx != last_idx + 1:
                    indices_sorted = False
                    break
                last_idx = idx
        all_chunk_indices.append(chunk_indices)

    if indices_sorted and not hasattr(funcs_args, "__len__"):
        # Iterate through funcs_args
        outputs = []
        chunk_idx = 0
        _funcs_args = []

        with get_pbar(total=len(all_chunk_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
            for i, func_args in enumerate(funcs_args):
                if i > all_chunk_indices[chunk_idx][-1]:
                    chunk_indices = all_chunk_indices[chunk_idx]
                    outputs.extend(_execute(_funcs_args, len(chunk_indices)))
                    chunk_idx += 1
                    _funcs_args = []
                    pbar.update(1)
                _funcs_args.append(func_args)
            if len(_funcs_args) > 0:
                chunk_indices = all_chunk_indices[chunk_idx]
                outputs.extend(_execute(_funcs_args, len(chunk_indices)))
                pbar.update(1)
        return outputs
    else:
        # Iterate through chunks
        funcs_args = list(funcs_args)
        outputs = []

        with get_pbar(total=len(all_chunk_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
            for chunk_indices in all_chunk_indices:
                _funcs_args = []
                for idx in chunk_indices:
                    _funcs_args.append(funcs_args[idx])
                chunk_output = _execute(_funcs_args, len(chunk_indices))
                if in_chunk_order or indices_sorted:
                    outputs.extend(chunk_output)
                else:
                    outputs.extend(zip(chunk_indices, chunk_output))
                pbar.update(1)
        if in_chunk_order or indices_sorted:
            return outputs
        return list(list(zip(*sorted(outputs, key=lambda x: x[0])))[1])
