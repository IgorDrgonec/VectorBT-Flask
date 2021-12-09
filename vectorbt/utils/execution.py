# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Engines for executing functions."""

from numba.core.registry import CPUDispatcher

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts, Configured
from vectorbt.utils.pbar import get_pbar

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

    For defaults, see `execution.engines.sequence` in `vectorbt._settings.settings`."""

    def __init__(self,
                 show_progress: tp.Optional[bool] = None,
                 pbar_kwargs: tp.KwargsLike = None) -> None:
        from vectorbt._settings import settings
        sequence_cfg = settings['execution']['engines']['sequence']

        if show_progress is None:
            show_progress = sequence_cfg['show_progress']
        pbar_kwargs = merge_dicts(pbar_kwargs, sequence_cfg['pbar_kwargs'])

        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

        ExecutionEngine.__init__(
            self,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs
        )

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar using `vectorbt.utils.pbar.get_pbar`."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbt.utils.pbar.get_pbar`."""
        return self._pbar_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        results = []
        with get_pbar(total=n_calls, show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for func, args, kwargs in funcs_args:
                results.append(func(*args, **kwargs))
                pbar.update(1)
        return results


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    For defaults, see `execution.engines.dask` in `vectorbt._settings.settings`.

    !!! note
        Use multi-threading mainly on numeric code that releases the GIL
        (like NumPy, Pandas, Scikit-Learn, Numba)."""

    def __init__(self, **compute_kwargs) -> None:
        from vectorbt._settings import settings
        dask_cfg = settings['execution']['engines']['dask']

        compute_kwargs = merge_dicts(compute_kwargs, dask_cfg['compute_kwargs'])

        self._compute_kwargs = compute_kwargs

        ExecutionEngine.__init__(
            self,
            **compute_kwargs
        )

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `dask.compute`."""
        return self._compute_kwargs

    def execute(self, funcs_args: tp.FuncsArgs, n_calls: tp.Optional[int] = None) -> list:
        from vectorbt.opt_packages import assert_can_import
        assert_can_import('dask')
        import dask

        results_delayed = []
        for func, args, kwargs in funcs_args:
            results_delayed.append(dask.delayed(func)(*args, **kwargs))
        return list(dask.compute(*results_delayed, **self.compute_kwargs))


class RayEngine(ExecutionEngine):
    """Class for executing functions in parallel using Ray.

    For defaults, see `execution.engines.ray` in `vectorbt._settings.settings`.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup."""

    def __init__(self,
                 restart: tp.Optional[bool] = None,
                 reuse_refs: tp.Optional[bool] = None,
                 del_refs: tp.Optional[bool] = None,
                 shutdown: tp.Optional[bool] = None,
                 init_kwargs: tp.KwargsLike = None,
                 remote_kwargs: tp.KwargsLike = None) -> None:
        from vectorbt._settings import settings
        ray_cfg = settings['execution']['engines']['ray']

        if restart is None:
            restart = ray_cfg['restart']
        if reuse_refs is None:
            reuse_refs = ray_cfg['reuse_refs']
        if del_refs is None:
            del_refs = ray_cfg['del_refs']
        if shutdown is None:
            shutdown = ray_cfg['shutdown']
        init_kwargs = merge_dicts(init_kwargs, ray_cfg['init_kwargs'])
        remote_kwargs = merge_dicts(remote_kwargs, ray_cfg['remote_kwargs'])

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
            remote_kwargs=remote_kwargs
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
    def get_ray_refs(funcs_args: tp.FuncsArgs,
                     reuse_refs: bool = True,
                     remote_kwargs: tp.KwargsLike = None) -> \
            tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Get result references by putting each argument and keyword argument into the object store
        and invoking the remote decorator on each function using Ray.

        If `reuse_refs` is True, will generate one reference per unique object id."""
        from vectorbt.opt_packages import assert_can_import
        assert_can_import('ray')
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
        from vectorbt.opt_packages import assert_can_import
        assert_can_import('ray')
        import ray

        if self.restart:
            if ray.is_initialized():
                ray.shutdown()
        if not ray.is_initialized():
            ray.init(**self.init_kwargs)
        funcs_args_refs = self.get_ray_refs(
            funcs_args,
            reuse_refs=self.reuse_refs,
            remote_kwargs=self.remote_kwargs
        )
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


def execute(funcs_args: tp.FuncsArgs,
            engine: tp.EngineLike = SequenceEngine,
            n_calls: tp.Optional[int] = None,
            **kwargs) -> list:
    """Execute using an engine.

    Supported values for `engine`:

    * Name of the engine (see supported engines)
    * Subclass of `ExecutionEngine` - will initialize with `**kwargs`
    * Instance of `ExecutionEngine` - will call `ExecutionEngine.execute` with `n_calls`
    * Callable - will pass `funcs_args`, `n_calls` (if not None), and `**kwargs`

    Supported engines can be found in `execution.engines` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    engines_cfg = settings['execution']['engines']

    if isinstance(engine, str):
        if engine.lower() in engines_cfg:
            engine = engines_cfg[engine]['cls']
        else:
            raise ValueError(f"Engine with name '{engine}' is unknown")
    if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
        engine = engine(**kwargs)
    if isinstance(engine, ExecutionEngine):
        return engine.execute(funcs_args, n_calls=n_calls)
    if callable(engine):
        if n_calls is not None:
            kwargs['n_calls'] = n_calls
        return engine(funcs_args, **kwargs)
    raise TypeError(f"Engine of type {type(engine)} is not supported")
