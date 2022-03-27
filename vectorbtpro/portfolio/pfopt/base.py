# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Base functions and classes for portfolio optimization."""

import inspect
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.config import merge_dicts, resolve_dict, Config, HybridConfig
from vectorbtpro.utils.template import deep_substitute, Rep, RepFunc
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.base.indexes import combine_indexes, stack_indexes
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array, to_dict
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.enums import RangeStatus
from vectorbtpro.portfolio.enums import alloc_range_dt, alloc_point_dt
from vectorbtpro.portfolio.pfopt import nb
from vectorbtpro.portfolio.pfopt.records import AllocRanges, AllocPoints
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg

try:
    from pypfopt.base_optimizer import BaseOptimizer

    BaseOptimizerT = tp.TypeVar("BaseOptimizerT", bound=BaseOptimizer)
except ImportError as e:
    BaseOptimizerT = tp.Any
try:
    from universal.algo import Algo
    from universal.result import AlgoResult

    AlgoT = tp.TypeVar("AlgoT", bound=Algo)
    AlgoResultT = tp.TypeVar("AlgoResultT", bound=AlgoResult)
except ImportError as e:
    AlgoT = tp.Any
    AlgoResultT = tp.Any


__pdoc__ = {}


# ############# PyPortfolioOpt ############# #


class pypfopt_func_dict(dict):
    """Dict that contains optimization functions as keys.

    Keys can be functions themselves, their names, or `_default` for the default value."""

    pass


def select_pypfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    kwargs: tp.Union[tp.Kwargs, pypfopt_func_dict],
) -> tp.Kwargs:
    """Select keyword arguments belonging to `pypfopt_func`."""
    if isinstance(kwargs, pypfopt_func_dict):
        if pypfopt_func in kwargs or pypfopt_func.__name__ in kwargs:
            _kwargs = kwargs[pypfopt_func]
        elif "_default" in kwargs:
            _kwargs = kwargs["_default"]
        else:
            _kwargs = {}
    else:
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, pypfopt_func_dict):
                if pypfopt_func in v or pypfopt_func.__name__ in v:
                    _kwargs[k] = v[pypfopt_func]
                elif "_default" in v:
                    _kwargs[k] = v["_default"]
            else:
                _kwargs[k] = v
    return _kwargs


def resolve_pypfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    cache: tp.KwargsLike = None,
    var_kwarg_names: tp.Optional[tp.Iterable[str]] = None,
    used_arg_names: tp.Optional[tp.Set[str]] = None,
    **kwargs,
) -> tp.Kwargs:
    """Resolve keyword arguments passed to any optimization function with the layout of PyPortfolioOpt.

    Parses the signature of `pypfopt_func`, and for each accepted argument, looks for an argument
    with the same name in `kwargs`. If not found, tries to resolve that argument using other arguments
    or by calling other optimization functions.

    Argument `frequency` gets resolved with (global) `freq` and `year_freq` using
    `vectorbtpro.returns.accessors.ReturnsAccessor.get_ann_factor`.

    Any argument in `kwargs` can be wrapped using `pypfopt_func_dict` to define the argument
    per function rather than globally.

    !!! note
        When providing custom functions, make sure that the arguments they accept are visible
        in the signature (that is, no variable arguments) and have the same naming as in PyPortfolioOpt.

        Functions `market_implied_prior_returns` and `BlackLittermanModel.bl_weights` take `risk_aversion`,
        which is different from arguments with the same name in other functions. To set it, pass `delta`."""
    from vectorbtpro.utils.opt_packages import assert_can_import

    assert_can_import("pypfopt")

    signature = inspect.signature(pypfopt_func)
    kwargs = select_pypfopt_func_kwargs(pypfopt_func, kwargs)
    if cache is None:
        cache = {}
    arg_names = get_func_arg_names(pypfopt_func)
    if len(arg_names) == 0:
        return {}
    if used_arg_names is None:
        used_arg_names = set()

    pass_kwargs = dict()

    def _process_arg(arg_name, arg_value):
        orig_arg_name = arg_name
        if pypfopt_func.__name__ in ("market_implied_prior_returns", "bl_weights"):
            if arg_name == "risk_aversion":
                # In some methods, risk_aversion is expected as array and means delta
                arg_name = "delta"

        def _get_kwarg(*args):
            used_arg_names.add(args[0])
            return kwargs.get(*args)

        def _get_prices():
            prices = None
            if "prices" in cache:
                prices = cache["prices"]
            elif "prices" in kwargs:
                if not _get_kwarg("returns_data", False):
                    prices = _get_kwarg("prices")
            return prices

        def _get_returns():
            returns = None
            if "returns" in cache:
                returns = cache["returns"]
            elif "returns" in kwargs:
                returns = _get_kwarg("returns")
            elif "prices" in kwargs and _get_kwarg("returns_data", False):
                returns = _get_kwarg("prices")
            return returns

        def _prices_from_returns():
            from pypfopt.expected_returns import prices_from_returns

            cache["prices"] = prices_from_returns(_get_returns(), _get_kwarg("log_returns", False))
            return cache["prices"]

        def _returns_from_prices():
            from pypfopt.expected_returns import returns_from_prices

            cache["returns"] = returns_from_prices(_get_prices(), _get_kwarg("log_returns", False))
            return cache["returns"]

        if arg_name == "expected_returns":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "expected_returns" not in cache:
                cache["expected_returns"] = resolve_pypfopt_expected_returns(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["expected_returns"]
        elif arg_name == "cov_matrix":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "cov_matrix" not in cache:
                cache["cov_matrix"] = resolve_pypfopt_cov_matrix(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["cov_matrix"]
        elif arg_name == "optimizer":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "optimizer" not in cache:
                cache["optimizer"] = resolve_pypfopt_optimizer(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["optimizer"]

        if orig_arg_name not in pass_kwargs:
            if arg_name in kwargs:
                if arg_name == "market_prices":
                    if pypfopt_func.__name__ != "market_implied_risk_aversion" and checks.is_series(
                        _get_kwarg(arg_name)
                    ):
                        pass_kwargs[orig_arg_name] = _get_kwarg(arg_name).to_frame().copy(deep=False)
                    else:
                        pass_kwargs[orig_arg_name] = _get_kwarg(arg_name).copy(deep=False)
                else:
                    pass_kwargs[orig_arg_name] = _get_kwarg(arg_name)
            else:
                if arg_name == "frequency":
                    ann_factor = ReturnsAccessor.get_ann_factor(_get_kwarg("year_freq", None), _get_kwarg("freq", None))
                    if ann_factor is not None:
                        pass_kwargs[orig_arg_name] = ann_factor
                elif arg_name == "prices":
                    if "returns_data" in arg_names:
                        if "returns_data" in kwargs:
                            if _get_kwarg("returns_data", False):
                                if _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _get_returns()
                                elif _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _returns_from_prices()
                            else:
                                if _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _get_prices()
                                elif _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _prices_from_returns()
                        else:
                            if _get_prices() is not None:
                                pass_kwargs[orig_arg_name] = _get_prices()
                                pass_kwargs["returns_data"] = False
                            elif _get_returns() is not None:
                                pass_kwargs[orig_arg_name] = _get_returns()
                                pass_kwargs["returns_data"] = True
                    else:
                        if _get_prices() is not None:
                            pass_kwargs[orig_arg_name] = _get_prices()
                        elif _get_returns() is not None:
                            pass_kwargs[orig_arg_name] = _prices_from_returns()
                elif arg_name == "returns":
                    if _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = _get_returns()
                    elif _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = _returns_from_prices()
                elif arg_name == "latest_prices":
                    from pypfopt.discrete_allocation import get_latest_prices

                    if _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(_get_prices())
                    elif _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(_prices_from_returns())
                elif arg_name == "delta":
                    if "delta" not in cache:
                        from pypfopt.black_litterman import market_implied_risk_aversion

                        cache["delta"] = resolve_pypfopt_func_call(
                            market_implied_risk_aversion,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["delta"]
                elif arg_name == "pi":
                    if "pi" not in cache:
                        from pypfopt.black_litterman import market_implied_prior_returns

                        cache["pi"] = resolve_pypfopt_func_call(
                            market_implied_prior_returns,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["pi"]

        if orig_arg_name not in pass_kwargs:
            if arg_value.default != inspect.Parameter.empty:
                pass_kwargs[orig_arg_name] = arg_value.default

    for arg_name, arg_value in signature.parameters.items():
        if arg_value.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"Variable positional arguments in {pypfopt_func} cannot be parsed")
        elif arg_value.kind == inspect.Parameter.VAR_KEYWORD:
            if var_kwarg_names is None:
                var_kwarg_names = []
            for var_arg_name in var_kwarg_names:
                _process_arg(var_arg_name, arg_value)
        else:
            _process_arg(arg_name, arg_value)

    return pass_kwargs


def resolve_pypfopt_func_call(pypfopt_func: tp.Callable, **kwargs) -> tp.Any:
    """Resolve arguments using `resolve_pypfopt_func_kwargs` and call the function with that arguments."""
    return pypfopt_func(**resolve_pypfopt_func_kwargs(pypfopt_func, **kwargs))


def resolve_pypfopt_expected_returns(
    expected_returns: tp.Union[tp.Callable, tp.AnyArray, str] = "mean_historical_return",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the expected returns.

    `expected_returns` can be an array, an attribute of `pypfopt.expected_returns`, a function,
    or one of the following options:

    * 'mean_historical_return': `pypfopt.expected_returns.mean_historical_return`
    * 'ema_historical_return': `pypfopt.expected_returns.ema_historical_return`
    * 'capm_return': `pypfopt.expected_returns.capm_return`
    * 'bl_returns': `pypfopt.black_litterman.BlackLittermanModel.bl_returns`

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.opt_packages import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(expected_returns, str):
        if expected_returns.lower() == "mean_historical_return":
            from pypfopt.expected_returns import mean_historical_return

            return resolve_pypfopt_func_call(mean_historical_return, **kwargs)
        if expected_returns.lower() == "ema_historical_return":
            from pypfopt.expected_returns import ema_historical_return

            return resolve_pypfopt_func_call(ema_historical_return, **kwargs)
        if expected_returns.lower() == "capm_return":
            from pypfopt.expected_returns import capm_return

            return resolve_pypfopt_func_call(capm_return, **kwargs)
        if expected_returns.lower() == "bl_returns":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            ).bl_returns()
        import pypfopt.expected_returns

        if hasattr(pypfopt.expected_returns, expected_returns):
            return resolve_pypfopt_func_call(getattr(pypfopt.expected_returns, expected_returns), **kwargs)
        raise NotImplementedError("Return model '{}' not supported".format(expected_returns))
    if callable(expected_returns):
        return resolve_pypfopt_func_call(expected_returns, **kwargs)
    return expected_returns


def resolve_pypfopt_cov_matrix(
    cov_matrix: tp.Union[tp.Callable, tp.AnyArray, str] = "ledoit_wolf",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the covariance matrix.

    `cov_matrix` can be an array, an attribute of `pypfopt.risk_models`, a function,
    or one of the following options:

    * 'sample_cov': `pypfopt.risk_models.sample_cov`
    * 'semicovariance' or 'semivariance': `pypfopt.risk_models.semicovariance`
    * 'exp_cov': `pypfopt.risk_models.exp_cov`
    * 'ledoit_wolf' or 'ledoit_wolf_constant_variance': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'constant_variance' as shrinkage factor
    * 'ledoit_wolf_single_factor': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'single_factor' as shrinkage factor
    * 'ledoit_wolf_constant_correlation': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'constant_correlation' as shrinkage factor
    * 'oracle_approximating': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'oracle_approximating' as shrinkage factor

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.opt_packages import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(cov_matrix, str):
        if cov_matrix.lower() == "sample_cov":
            from pypfopt.risk_models import sample_cov

            return resolve_pypfopt_func_call(sample_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "semicovariance" or cov_matrix.lower() == "semivariance":
            from pypfopt.risk_models import semicovariance

            return resolve_pypfopt_func_call(semicovariance, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "exp_cov":
            from pypfopt.risk_models import exp_cov

            return resolve_pypfopt_func_call(exp_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "ledoit_wolf" or cov_matrix.lower() == "ledoit_wolf_constant_variance":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf()
        if cov_matrix.lower() == "ledoit_wolf_single_factor":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="single_factor"
            )
        if cov_matrix.lower() == "ledoit_wolf_constant_correlation":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="constant_correlation"
            )
        if cov_matrix.lower() == "oracle_approximating":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).oracle_approximating()
        import pypfopt.risk_models

        if hasattr(pypfopt.risk_models, cov_matrix):
            return resolve_pypfopt_func_call(getattr(pypfopt.risk_models, cov_matrix), **kwargs)
        raise NotImplementedError("Risk model '{}' not supported".format(cov_matrix))
    if callable(cov_matrix):
        return resolve_pypfopt_func_call(cov_matrix, **kwargs)
    return cov_matrix


def resolve_pypfopt_optimizer(
    optimizer: tp.Union[tp.Callable, BaseOptimizerT, str] = "efficient_frontier",
    **kwargs,
) -> BaseOptimizerT:
    """Resolve the optimizer.

    `optimizer` can be an instance of `pypfopt.base_optimizer.BaseOptimizer`, an attribute of `pypfopt`,
    a subclass of  `pypfopt.base_optimizer.BaseOptimizer`, or one of the following options:

    * 'efficient_frontier': `pypfopt.efficient_frontier.EfficientFrontier`
    * 'efficient_cdar': `pypfopt.efficient_frontier.EfficientCDaR`
    * 'efficient_cvar': `pypfopt.efficient_frontier.EfficientCVaR`
    * 'efficient_semivariance': `pypfopt.efficient_frontier.EfficientSemivariance`
    * 'black_litterman' or 'bl': `pypfopt.black_litterman.BlackLittermanModel`
    * 'hierarchical_portfolio', 'hrpopt', or 'hrp': `pypfopt.hierarchical_portfolio.HRPOpt`
    * 'cla': `pypfopt.cla.CLA`

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.opt_packages import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(optimizer, str):
        if optimizer.lower() == "efficient_frontier":
            from pypfopt.efficient_frontier import EfficientFrontier

            return resolve_pypfopt_func_call(EfficientFrontier, **kwargs)
        if optimizer.lower() == "efficient_cdar":
            from pypfopt.efficient_frontier import EfficientCDaR

            return resolve_pypfopt_func_call(EfficientCDaR, **kwargs)
        if optimizer.lower() == "efficient_cvar":
            from pypfopt.efficient_frontier import EfficientCVaR

            return resolve_pypfopt_func_call(EfficientCVaR, **kwargs)
        if optimizer.lower() == "efficient_semivariance":
            from pypfopt.efficient_frontier import EfficientSemivariance

            return resolve_pypfopt_func_call(EfficientSemivariance, **kwargs)
        if optimizer.lower() == "black_litterman" or optimizer.lower() == "bl":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            )
        if optimizer.lower() == "hierarchical_portfolio" or optimizer.lower() == "hrpopt" or optimizer.lower() == "hrp":
            from pypfopt.hierarchical_portfolio import HRPOpt

            return resolve_pypfopt_func_call(HRPOpt, **kwargs)
        if optimizer.lower() == "cla":
            from pypfopt.cla import CLA

            return resolve_pypfopt_func_call(CLA, **kwargs)
        import pypfopt

        if hasattr(pypfopt, optimizer):
            return resolve_pypfopt_func_call(getattr(pypfopt, optimizer), **kwargs)
        raise NotImplementedError("Optimizer '{}' not supported".format(optimizer))
    if isinstance(optimizer, type) and issubclass(optimizer, BaseOptimizer):
        return resolve_pypfopt_func_call(optimizer, **kwargs)
    if isinstance(optimizer, BaseOptimizer):
        return optimizer
    raise NotImplementedError("Optimizer {} not supported".format(optimizer))


def pypfopt_optimize(
    target: tp.Optional[tp.Union[tp.Callable, str]] = None,
    target_is_convex: tp.Optional[bool] = None,
    weights_sum_to_one: tp.Optional[bool] = None,
    target_constraints: tp.Optional[tp.List[tp.Kwargs]] = None,
    target_solver: tp.Optional[str] = None,
    target_initial_guess: tp.Optional[tp.Array] = None,
    objectives: tp.Optional[tp.MaybeIterable[tp.Union[tp.Callable, str]]] = None,
    constraints: tp.Optional[tp.MaybeIterable[tp.Callable]] = None,
    sector_mapper: tp.Optional[dict] = None,
    sector_lower: tp.Optional[dict] = None,
    sector_upper: tp.Optional[dict] = None,
    discrete_allocation: tp.Optional[bool] = None,
    allocation_method: tp.Optional[str] = None,
    silence_warnings: tp.Optional[bool] = None,
    ignore_opt_errors: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Dict[str, float]:
    """Get allocation using PyPortfolioOpt.

    First, it resolves the optimizer using `resolve_pypfopt_optimizer`. Depending upon which arguments it takes,
    it may further resolve expected returns, covariance matrix, etc. Then, it adds objectives and constraints
    to the optimizer instance, calls the target metric, extracts the weights, and finally, converts
    the weights to an integer allocation (if requested).

    To specify the optimizer, use `optimizer` (see `resolve_pypfopt_optimizer`).
    To specify the expected returns, use `expected_returns` (see `resolve_pypfopt_expected_returns`).
    To specify the covariance matrix, use `cov_matrix` (see `resolve_pypfopt_cov_matrix`).
    All other keyword arguments in `**kwargs` are used by `resolve_pypfopt_func_call`.

    Each objective can be a function, an attribute of `pypfopt.objective_functions`, or an iterable of such.

    Each constraint can be a function or an interable of such.

    The target can be an attribute of the optimizer, or a stand-alone function.
    If `target_is_convex` is True, the function is added as a convex function.
    Otherwise, the function is added as a non-convex function. The keyword arguments
    `weights_sum_to_one` and those starting with `target` are passed
    `pypfopt.base_optimizer.BaseConvexOptimizer.convex_objective`
    and `pypfopt.base_optimizer.BaseConvexOptimizer.nonconvex_objective` respectively.
    Set `ignore_opt_errors` to True to ignore any target optimization errors.

    If `discrete_allocation` is True, resolves `pypfopt.discrete_allocation.DiscreteAllocation`
    and calls `allocation_method` as an attribute of the allocation object.

    Any function is resolved using `resolve_pypfopt_func_call`.

    For defaults, see `vectorbtpro._settings.pfopt`.

    Usage:
        * Using mean historical returns, Ledoit-Wolf covariance matrix with constant variance,
        and efficient frontier:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> from vectorbtpro.portfolio.pfopt.base import pypfopt_optimize

        >>> data = vbt.YFData.fetch(["MSFT", "AMZN", "KO", "MA"])
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> pypfopt_optimize(prices=data.get("Close"))
        OrderedDict([('MSFT', 0.13082),
                     ('AMZN', 0.10451),
                     ('KO', 0.02513),
                     ('MA', 0.73954)])
        ```

        * EMA historical returns and sample covariance:

        ```pycon
        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     cov_matrix="sample_cov"
        ... )
        OrderedDict([('MSFT', 0.46812), ('AMZN', 0.0), ('KO', 0.53188), ('MA', 0.0)])
        ```

        * EMA historical returns, efficient Conditional Value at Risk, and other parameters automatically
        passed to their respective functions. Optimized towards lowest CVaR:

        ```pycon
        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     optimizer="efficient_cvar",
        ...     beta=0.9,
        ...     weight_bounds=(-1, 1),
        ...     target="min_cvar"
        ... )
        OrderedDict([('MSFT', 0.14679),
                     ('AMZN', 0.08134),
                     ('KO', 0.76805),
                     ('MA', 0.00382)])
        ```

        * Adding custom objectives:

        ```pycon
        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     objectives=["L2_reg"],
        ...     gamma=0.1,
        ...     target="min_volatility"
        ... )
        OrderedDict([('MSFT', 0.22246),
                     ('AMZN', 0.15801),
                     ('KO', 0.28516),
                     ('MA', 0.33437)])
        ```

        * Adding custom constraints:

        ```pycon
        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     constraints=[lambda w: w[data.symbols.index("MSFT")] <= 0.1]
        ... )
        OrderedDict([('MSFT', 0.1),
                     ('AMZN', 0.11061),
                     ('KO', 0.03545),
                     ('MA', 0.75394)])
        ```

        * Optimizing towards a custom convex objective (to add a non-convex objective,
        set `target_is_convex` to False):

        ```pycon
        >>> import cvxpy as cp

        >>> def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
        ...     log_sum = cp.sum(cp.log(w))
        ...     var = cp.quad_form(w, cov_matrix)
        ...     return var - k * log_sum

        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     target=logarithmic_barrier_objective
        ... )
        OrderedDict([('MSFT', 0.24599),
                     ('AMZN', 0.23072),
                     ('KO', 0.25823),
                     ('MA', 0.26507)])
        ```
    """
    from vectorbtpro.utils.opt_packages import assert_can_import

    assert_can_import("pypfopt")
    from pypfopt.exceptions import OptimizationError
    from cvxpy.error import SolverError

    from vectorbtpro._settings import settings

    pfopt_cfg = dict(settings["pfopt"])

    def _resolve_setting(k, v):
        setting = pfopt_cfg.pop(k)
        if v is None:
            return setting
        return v

    target = _resolve_setting("target", target)
    target_is_convex = _resolve_setting("target_is_convex", target_is_convex)
    weights_sum_to_one = _resolve_setting("weights_sum_to_one", weights_sum_to_one)
    target_constraints = _resolve_setting("target_constraints", target_constraints)
    target_solver = _resolve_setting("target_solver", target_solver)
    target_initial_guess = _resolve_setting("target_initial_guess", target_initial_guess)
    objectives = _resolve_setting("objectives", objectives)
    constraints = _resolve_setting("constraints", constraints)
    sector_mapper = _resolve_setting("sector_mapper", sector_mapper)
    sector_lower = _resolve_setting("sector_lower", sector_lower)
    sector_upper = _resolve_setting("sector_upper", sector_upper)
    discrete_allocation = _resolve_setting("discrete_allocation", discrete_allocation)
    allocation_method = _resolve_setting("allocation_method", allocation_method)
    silence_warnings = _resolve_setting("silence_warnings", silence_warnings)
    ignore_opt_errors = _resolve_setting("ignore_opt_errors", ignore_opt_errors)
    kwargs = merge_dicts(pfopt_cfg, kwargs)

    if "cache" not in kwargs:
        kwargs["cache"] = {}
    if "used_arg_names" not in kwargs:
        kwargs["used_arg_names"] = set()

    optimizer = kwargs["optimizer"] = resolve_pypfopt_optimizer(**kwargs)

    if objectives is not None:
        if not checks.is_iterable(objectives) or isinstance(objectives, str):
            objectives = [objectives]
        for objective in objectives:
            if isinstance(objective, str):
                import pypfopt.objective_functions

                objective = getattr(pypfopt.objective_functions, objective)
            objective_kwargs = resolve_pypfopt_func_kwargs(objective, **kwargs)
            optimizer.add_objective(objective, **objective_kwargs)
    if constraints is not None:
        if not checks.is_iterable(constraints):
            constraints = [constraints]
        for constraint in constraints:
            optimizer.add_constraint(constraint)
    if sector_mapper is not None:
        if sector_lower is None:
            sector_lower = {}
        if sector_upper is None:
            sector_upper = {}
        optimizer.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    try:
        if isinstance(target, str):
            resolve_pypfopt_func_call(getattr(optimizer, target), **kwargs)
        else:
            if target_is_convex:
                optimizer.convex_objective(
                    target,
                    weights_sum_to_one=weights_sum_to_one,
                    **resolve_pypfopt_func_kwargs(target, **kwargs),
                )
            else:
                optimizer.nonconvex_objective(
                    target,
                    objective_args=tuple(resolve_pypfopt_func_kwargs(target, **kwargs).values()),
                    weights_sum_to_one=weights_sum_to_one,
                    constraints=target_constraints,
                    solver=target_solver,
                    initial_guess=target_initial_guess,
                )
    except (OptimizationError, SolverError) as e:
        if ignore_opt_errors:
            return {}
        raise e

    weights = kwargs["weights"] = resolve_pypfopt_func_call(optimizer.clean_weights, **kwargs)
    if discrete_allocation:
        from pypfopt.discrete_allocation import DiscreteAllocation

        allocator = resolve_pypfopt_func_call(DiscreteAllocation, **kwargs)
        return resolve_pypfopt_func_call(getattr(allocator, allocation_method), **kwargs)[0]

    passed_arg_names = set(kwargs.keys())
    passed_arg_names.remove("cache")
    passed_arg_names.remove("used_arg_names")
    passed_arg_names.remove("optimizer")
    passed_arg_names.remove("weights")
    unused_arg_names = passed_arg_names.difference(kwargs["used_arg_names"])
    if len(unused_arg_names) > 0:
        if not silence_warnings:
            warnings.warn(f"Some arguments were not used: {unused_arg_names}", stacklevel=2)

    return weights


# ############# PortfolioOptimizer ############# #


class pfopt_group_dict(dict):
    """Dict that contains optimization groups as keys.

    Keys can be group identifiers or `_default` for the default value."""

    pass


def find_pfopt_groups(
    args: tp.Union[tp.Args, pypfopt_func_dict],
    kwargs: tp.Union[tp.Kwargs, pypfopt_func_dict],
    sort_groups: bool = False,
) -> tp.List[tp.Hashable]:
    """Find all groups in arguments."""
    groups = []
    if isinstance(args, pfopt_group_dict):
        for group in args:
            if group not in groups and group != "_default":
                groups.append(group)
    else:
        for arg in args:
            if isinstance(arg, pfopt_group_dict):
                for group in arg:
                    if group not in groups and group != "_default":
                        groups.append(group)
    if isinstance(kwargs, pfopt_group_dict):
        for group in kwargs:
            if group not in groups and group != "_default":
                groups.append(group)
    else:
        for k, v in kwargs.items():
            if isinstance(v, pfopt_group_dict):
                for group in v:
                    if group not in groups and group != "_default":
                        groups.append(group)
    if sort_groups:
        return sorted(groups)
    return groups


def select_pfopt_group_args(
    group: tp.Hashable,
    args: tp.Union[tp.Args, pypfopt_func_dict],
    kwargs: tp.Union[tp.Kwargs, pypfopt_func_dict],
) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Select arguments and keyword arguments belonging to `group`.

    If an instance of `pfopt_group_dict` was found and neither the group nor the default (`_default`)
    are present, ignores the argument."""
    if isinstance(args, pfopt_group_dict):
        if group in args:
            _args = args[group]
        elif "_default" in args:
            _args = args["_default"]
        else:
            _args = ()
    else:
        _args = ()
        for v in args:
            if isinstance(v, pfopt_group_dict):
                if group in v:
                    _args += (v[group],)
                elif "_default" in v:
                    _args += (v["_default"],)
            else:
                _args += (v,)
    if isinstance(kwargs, pfopt_group_dict):
        if group in kwargs:
            _kwargs = kwargs[group]
        elif "_default" in kwargs:
            _kwargs = kwargs["_default"]
        else:
            _kwargs = {}
    else:
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, pfopt_group_dict):
                if group in v:
                    _kwargs[k] = v[group]
                elif "_default" in v:
                    _kwargs[k] = v["_default"]
            else:
                _kwargs[k] = v
    return _args, _kwargs


PortfolioOptimizerT = tp.TypeVar("PortfolioOptimizerT", bound="PortfolioOptimizer")


class PortfolioOptimizer(Analyzable):
    """Class that exposes methods for generating allocations."""

    def __init__(
        self,
        wrapper: ArrayWrapper,
        alloc_records: tp.Union[AllocRanges, AllocPoints],
        allocations: tp.Array2d,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            alloc_records=alloc_records,
            allocations=allocations,
            **kwargs,
        )

        self._alloc_records = alloc_records
        self._allocations = allocations

        # Cannot select rows
        self._column_only_select = True

    def indexing_func(
        self: PortfolioOptimizerT,
        pd_indexing_func: tp.PandasIndexingFunc,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Perform indexing on `PortfolioOptimizer`."""
        new_wrapper, _, group_idxs, _ = self._wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
        new_alloc_records = self._alloc_records.indexing_func(pd_indexing_func, **kwargs)
        new_indices, _ = self._alloc_records.col_mapper.select_cols(group_idxs)
        new_allocations = to_2d_array(self._allocations)[new_indices]
        return self.replace(wrapper=new_wrapper, alloc_records=new_alloc_records, allocations=new_allocations)

    def resample(self: PortfolioOptimizerT, *args, bfill_ts: bool = False, **kwargs) -> PortfolioOptimizerT:
        """Perform resampling on `PortfolioOptimizer`."""
        new_wrapper = self._wrapper.resample(*args, **kwargs)
        new_alloc_records = self._alloc_records.resample(*args, bfill_ts=bfill_ts, **kwargs)
        return self.replace(
            wrapper=new_wrapper,
            alloc_records=new_alloc_records,
        )

    # ############# Class methods ############# #

    @classmethod
    def from_optimize_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        optimize_func: tp.Union[tp.Callable, pfopt_group_dict],
        *args,
        jitted_loop: tp.Union[bool, pfopt_group_dict] = False,
        every: tp.Union[None, tp.FrequencyLike, pfopt_group_dict] = None,
        normalize_every: tp.Union[bool, pfopt_group_dict] = False,
        lookback_period: tp.Union[None, tp.FrequencyLike, pfopt_group_dict] = None,
        start: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, pfopt_group_dict] = None,
        end: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, pfopt_group_dict] = None,
        closed_start: tp.Union[bool, pfopt_group_dict] = True,
        closed_end: tp.Union[bool, pfopt_group_dict] = False,
        kind: tp.Union[None, str, pfopt_group_dict] = None,
        bounds_skipna: tp.Union[bool, pfopt_group_dict] = True,
        jitted: tp.Union[tp.JittedOption, pfopt_group_dict] = None,
        chunked: tp.Union[tp.ChunkedOption, pfopt_group_dict] = None,
        index_ranges: tp.Union[None, tp.MaybeSequence[tp.MaybeSequence[int]], pfopt_group_dict] = None,
        index_loc: tp.Union[None, tp.MaybeSequence[int], pfopt_group_dict] = None,
        alloc_wait: tp.Union[int, pfopt_group_dict] = 1,
        groups: tp.Optional[tp.Sequence[tp.Hashable]] = None,
        template_context: tp.Union[None, tp.Kwargs, pfopt_group_dict] = None,
        execute_kwargs: tp.Union[None, tp.Kwargs, pfopt_group_dict] = None,
        sort_groups: bool = False,
        wrapper_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        forward_args: tp.Optional[tp.Sequence[str]] = None,
        forward_kwargs: tp.Optional[tp.Sequence[str]] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an optimization function.

        Generates date ranges, performs optimization on the subset of data that belongs to each date range,
        and allocates at the end of each range.

        This is a parametrized method that allows testing multiple combinations on most arguments.
        First, it uses `find_pfopt_groups` to check whether any of the arguments is wrapped
        with `pfopt_group_dict` and extracts the keys of all groups that were found. After that,
        it iterates over each group (= parameter combination), and selects all arguments and
        keyword arguments that correspond to that group using `select_pfopt_group_args`.

        It then resolves the date ranges, either using the ready-to-use `index_ranges` or
        by passing all the arguments ranging from `every` to `jitted` to
        `vectorbtpro.base.wrapping.ArrayWrapper.get_index_ranges`. The optimization
        function `optimize_func` is then called on each date range by first substituting
        any templates found in `*args` and `**kwargs`. To forward any reserved arguments
        such as `jitted` to the optimization function, specify their names in `forward_args`
        and `forward_kwargs`.

        !!! note
            Make sure to use vectorbt's own templates to select the current date range
            (available as `index_slice` in the context mapping) from each array.

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.optimize_meta_nb`.
        Otherwise, must take template-substituted `*args` and `**kwargs`, and return an array or
        dictionary with asset allocations (also empty).

        !!! note
            When `jitted_loop` is True and in case of multiple groups, use templates
            to substitute by the current group index (available as `group_idx` in the context mapping).

        All allocations of all groups are stacked into one big 2-dim array where columns are assets
        and rows are allocations. Furthermore, date ranges are used to fill a record array of type
        `vectorbtpro.portfolio.pfopt.records.AllocRanges` that acts as an indexer for allocations.
        For example, the field `col` stores the group index corresponding to each allocation. Since
        this record array does not hold any information on assets themselves, it has its own wrapper
        that holds groups instead of columns, while the wrapper of the `PortfolioOptimizer` instance
        contains regular columns grouped by groups.

        Usage:
            * Allocate once:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> data = vbt.YFData.fetch(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def optimize_func(df):
            ...     sharpe = df.mean() / df.std()
            ...     return sharpe / sharpe.sum()

            >>> df_arg = vbt.RepEval("close.iloc[index_slice]", context=dict(close=close))
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     end="2015-01-01"
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2015-01-02 00:00:00+00:00  0.402459  0.309351  0.288191
            ```

            * Allocate every first date of the year:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN"
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2011-01-03 00:00:00+00:00  0.480693  0.257317  0.261990
                        2012-01-03 00:00:00+00:00  0.489893  0.215381  0.294727
                        2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                        2014-01-02 00:00:00+00:00  0.339649  0.273996  0.386354
                        2015-01-02 00:00:00+00:00  0.350406  0.418638  0.230956
                        2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                        2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                        2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                        2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            ```

            * Specify index ranges manually:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ]
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2010-02-16 00:00:00+00:00  0.340641  0.285897  0.373462
                        2010-03-30 00:00:00+00:00  0.596392  0.206317  0.197291
                        2010-05-12 00:00:00+00:00  0.437481  0.283160  0.279358
            ```

            * Test multiple combinations of one argument:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN",
            ...     start="2015-01-01",
            ...     lookback_period=vbt.pfopt_group_dict({
            ...         "3MS": "3MS",
            ...         "6MS": "6MS"
            ...     })
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            3MS         2016-01-04 00:00:00+00:00  0.282725  0.234970  0.482305
                        2017-01-03 00:00:00+00:00  0.318100  0.269355  0.412545
                        2018-01-02 00:00:00+00:00  0.387499  0.236432  0.376068
                        2019-01-02 00:00:00+00:00  0.575464  0.254808  0.169728
            6MS         2016-01-04 00:00:00+00:00  0.265035  0.198619  0.536346
                        2017-01-03 00:00:00+00:00  0.314144  0.409020  0.276836
                        2018-01-02 00:00:00+00:00  0.322741  0.282639  0.394621
                        2019-01-02 00:00:00+00:00  0.565691  0.234760  0.199549
            ```

            * Test multiple cross-argument combinations:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every=vbt.pfopt_group_dict({
            ...         1: "MS",
            ...         "_default": "AS-JAN"
            ...     }),
            ...     start=vbt.pfopt_group_dict({
            ...         0: "2015-01-01",
            ...         1: "2019-06-01"
            ...     }),
            ...     end=vbt.pfopt_group_dict({
            ...         2: "2014-01-01"
            ...     }),
            ...     sort_groups=True
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            0           2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                        2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                        2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                        2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            1           2019-07-01 00:00:00+00:00  0.351462  0.327333  0.321205
                        2019-08-01 00:00:00+00:00  0.418411  0.249798  0.331790
                        2019-09-03 00:00:00+00:00  0.400439  0.374044  0.225516
                        2019-10-01 00:00:00+00:00  0.509386  0.250497  0.240117
                        2019-11-01 00:00:00+00:00  0.349984  0.469181  0.180835
                        2019-12-02 00:00:00+00:00  0.260436  0.380564  0.359000
            2           2011-01-03 00:00:00+00:00  0.480693  0.257317  0.261990
                        2012-01-03 00:00:00+00:00  0.489893  0.215381  0.294727
                        2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                        2014-01-02 00:00:00+00:00  0.339649  0.273996  0.386354
            ```

            * Use Numba-compiled loop:

            ```pycon
            >>> from numba import njit
            >>> import numpy as np

            >>> @njit
            ... def optimize_func_nb(i, from_idx, to_idx, close):
            ...     mean = vbt.nb.nanmean_nb(close[from_idx:to_idx])
            ...     std = vbt.nb.nanstd_nb(close[from_idx:to_idx])
            ...     sharpe = mean / std
            ...     return sharpe / np.sum(sharpe)

            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func_nb,
            ...     np.asarray(close),
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ],
            ...     jitted_loop=True
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2010-02-16 00:00:00+00:00  0.340641  0.285897  0.373462
                        2010-03-30 00:00:00+00:00  0.596392  0.206317  0.197291
                        2010-05-12 00:00:00+00:00  0.437481  0.283160  0.279358
            ```

            !!! hint
                There is no big reason of using the Numba-compiled loop, apart from when having
                to rebalance many thousands of times. Usually, using a regular Python loop
                and a Numba-compiled optimization function suffice.
        """
        if pbar_kwargs is None:
            pbar_kwargs = {}
        if isinstance(wrapper.columns, pd.MultiIndex):
            raise ValueError("Columns must represent only assets and cannot be multi-index")

        groupable_kwargs = {
            "optimize_func": optimize_func,
            "jitted_loop": jitted_loop,
            "every": every,
            "normalize_every": normalize_every,
            "lookback_period": lookback_period,
            "start": start,
            "end": end,
            "closed_start": closed_start,
            "closed_end": closed_end,
            "kind": kind,
            "bounds_skipna": bounds_skipna,
            "jitted": jitted,
            "chunked": chunked,
            "index_ranges": index_ranges,
            "index_loc": index_loc,
            "alloc_wait": alloc_wait,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
            "forward_args": forward_args,
            "forward_kwargs": forward_kwargs,
            **kwargs,
        }
        if groups is None:
            groups = find_pfopt_groups(args, groupable_kwargs, sort_groups=sort_groups)
        if len(groups) == 0:
            groups = ["group"]
            single_group = True
        else:
            single_group = False

        alloc_ranges = []
        allocations = []
        if show_progress is None:
            show_progress = len(groups) > 1
            show_progress_none = True
        else:
            show_progress_none = False
        with get_pbar(total=len(groups), show_progress=show_progress, **pbar_kwargs) as pbar:
            for g, group in enumerate(groups):
                pbar.set_description(str(group))

                _args, _kwargs = select_pfopt_group_args(group, args, groupable_kwargs)
                _optimize_func = _kwargs.pop("optimize_func")
                _jitted_loop = _kwargs.pop("jitted_loop", False)
                _every = _kwargs.pop("every", None)
                _normalize_every = _kwargs.pop("normalize_every", False)
                _lookback_period = _kwargs.pop("lookback_period", None)
                _start = _kwargs.pop("start", None)
                _end = _kwargs.pop("end", None)
                _closed_start = _kwargs.pop("closed_start", True)
                _closed_end = _kwargs.pop("closed_end", False)
                _kind = _kwargs.pop("kind", None)
                _bounds_skipna = _kwargs.pop("bounds_skipna", True)
                _jitted = _kwargs.pop("jitted", None)
                _chunked = _kwargs.pop("chunked", None)
                _index_ranges = _kwargs.pop("index_ranges", None)
                _index_loc = _kwargs.pop("index_loc", None)
                _alloc_wait = _kwargs.pop("alloc_wait", 1)
                _template_context = _kwargs.pop("template_context", None)
                _execute_kwargs = _kwargs.pop("execute_kwargs", None)
                _forward_args = _kwargs.pop("forward_args", None)
                _forward_kwargs = _kwargs.pop("forward_kwargs", None)

                _template_context = merge_dicts(
                    dict(
                        groups=groups,
                        group=group,
                        group_idx=g,
                        wrapper=wrapper,
                        optimize_func=_optimize_func,
                        jitted_loop=_jitted_loop,
                        every=_every,
                        normalize_every=_normalize_every,
                        lookback_period=_lookback_period,
                        start=_start,
                        end=_end,
                        closed_start=_closed_start,
                        closed_end=_closed_end,
                        kind=_kind,
                        bounds_skipna=_bounds_skipna,
                        jitted=_jitted,
                        chunked=_chunked,
                        index_ranges=_index_ranges,
                        index_loc=_index_loc,
                        args=_args,
                        kwargs=_kwargs,
                        execute_kwargs=_execute_kwargs,
                        forward_args=_forward_args,
                        forward_kwargs=_forward_kwargs,
                    ),
                    _template_context,
                )

                if _index_ranges is None:
                    get_index_ranges_kwargs = deep_substitute(
                        dict(
                            every=_every,
                            normalize_every=_normalize_every,
                            lookback_period=_lookback_period,
                            start=_start,
                            end=_end,
                            closed_start=_closed_start,
                            closed_end=_closed_end,
                            kind=_kind,
                            skipna=_bounds_skipna,
                            jitted=_jitted,
                        ),
                        _template_context,
                        sub_id="get_index_ranges_kwargs",
                        strict=True,
                    )
                    _index_ranges = wrapper.get_index_ranges(**get_index_ranges_kwargs)
                    _template_context = merge_dicts(
                        _template_context,
                        get_index_ranges_kwargs,
                        dict(index_ranges=_index_ranges),
                    )
                else:
                    _index_ranges = deep_substitute(
                        _index_ranges,
                        _template_context,
                        sub_id="index_ranges",
                        strict=True,
                    )
                    _index_ranges = to_2d_array(_index_ranges, expand_axis=0)
                if _index_loc is not None:
                    _index_loc = deep_substitute(
                        _index_loc,
                        _template_context,
                        sub_id="index_loc",
                        strict=True,
                    )
                    _index_loc = to_1d_array(_index_loc)

                if _forward_args is None:
                    _forward_args = []
                for k in _forward_args:
                    _args += (_template_context[k],)
                if _forward_kwargs is None:
                    _forward_kwargs = []
                for k in _forward_kwargs:
                    _kwargs[k] = _template_context[k]

                if jitted_loop:
                    _optimize_func = deep_substitute(
                        _optimize_func,
                        _template_context,
                        sub_id="optimize_func",
                        strict=True,
                    )
                    _args = deep_substitute(
                        _args,
                        _template_context,
                        sub_id="args",
                        strict=True,
                    )
                    _kwargs = deep_substitute(
                        _kwargs,
                        _template_context,
                        sub_id="kwargs",
                        strict=True,
                    )
                    func = jit_reg.resolve_option(nb.optimize_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    _allocations = func(len(wrapper.columns), _index_ranges, _optimize_func, *_args, **_kwargs)
                else:
                    funcs_args = []
                    for i in range(len(_index_ranges)):
                        index_slice = slice(max(0, _index_ranges[i, 0]), _index_ranges[i, 1])
                        __template_context = merge_dicts(dict(i=i, index_slice=index_slice), _template_context)
                        __optimize_func = deep_substitute(_optimize_func, _template_context, sub_id="optimize_func")
                        __args = deep_substitute(_args, __template_context, sub_id="args")
                        __kwargs = deep_substitute(_kwargs, __template_context, sub_id="kwargs")
                        funcs_args.append((__optimize_func, __args, __kwargs))

                    _execute_kwargs = merge_dicts(
                        dict(
                            show_progress=len(funcs_args) > 1 if show_progress_none else show_progress,
                            pbar_kwargs=pbar_kwargs,
                        ),
                        _execute_kwargs,
                    )
                    results = execute(funcs_args, **_execute_kwargs)
                    _allocations = pd.DataFrame(results, columns=wrapper.columns)
                    if isinstance(_allocations.columns, pd.RangeIndex):
                        _allocations = _allocations.values
                    else:
                        _allocations = _allocations[list(wrapper.columns)].values

                _alloc_ranges = np.empty(len(_allocations), alloc_range_dt)
                if _index_loc is None:
                    alloc_idx = _index_ranges[:, 1] - 1 + _alloc_wait
                else:
                    alloc_idx = _index_loc
                _alloc_ranges["id"] = np.arange(len(_allocations))
                _alloc_ranges["col"] = g
                _alloc_ranges["start_idx"] = _index_ranges[:, 0]
                _alloc_ranges["end_idx"] = _index_ranges[:, 1]
                _alloc_ranges["alloc_idx"] = alloc_idx
                _alloc_ranges["status"] = np.where(
                    alloc_idx >= len(wrapper.index),
                    RangeStatus.Open,
                    RangeStatus.Closed,
                )
                alloc_ranges.append(_alloc_ranges)
                allocations.append(_allocations)

                pbar.update(1)

        if isinstance(groups, pd.Index):
            group_index = groups
        else:
            group_index = pd.Index(groups, name="alloc_group")
        if group_index.has_duplicates:
            raise ValueError("Groups cannot have duplicates")
        if group_index.name is None:
            raise ValueError("Group index must have a name")
        new_columns = combine_indexes((group_index, wrapper.columns))
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_ranges = AllocRanges(
            ArrayWrapper(
                index=wrapper.index,
                columns=group_index,
                ndim=1 if single_group else 2,
                freq=wrapper.freq,
                column_only_select=True,
            ),
            np.concatenate(alloc_ranges),
        )
        allocations = np.row_stack(allocations)
        return cls(new_wrapper, alloc_ranges, allocations)

    @classmethod
    def from_pypfopt(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """`PortfolioOptimizer.from_optimize_func` applied on `pypfopt_optimize`."""
        if wrapper is None:
            if "prices" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["prices"])
            elif "returns" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["returns"])
            else:
                raise TypeError("Must provide a wrapper if price and returns are not set")
        return cls.from_optimize_func(wrapper, pypfopt_optimize, **kwargs)

    @classmethod
    def from_allocate_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocate_func: tp.Union[tp.Callable, pfopt_group_dict],
        *args,
        jitted_loop: tp.Union[bool, pfopt_group_dict] = False,
        every: tp.Union[None, tp.FrequencyLike, pfopt_group_dict] = None,
        normalize_every: tp.Union[bool, pfopt_group_dict] = False,
        start: tp.Union[None, int, tp.DatetimeLike, pfopt_group_dict] = None,
        end: tp.Union[None, int, tp.DatetimeLike, pfopt_group_dict] = None,
        on: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, pfopt_group_dict] = None,
        kind: tp.Union[None, str, pfopt_group_dict] = None,
        jitted: tp.Union[tp.JittedOption, pfopt_group_dict] = None,
        chunked: tp.Union[tp.ChunkedOption, pfopt_group_dict] = None,
        index_points: tp.Union[None, tp.MaybeSequence[int], pfopt_group_dict] = None,
        groups: tp.Optional[tp.Sequence[tp.Hashable]] = None,
        template_context: tp.Union[None, tp.Kwargs, pfopt_group_dict] = None,
        execute_kwargs: tp.Union[None, tp.Kwargs, pfopt_group_dict] = None,
        sort_groups: bool = False,
        wrapper_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        forward_args: tp.Optional[tp.Sequence[str]] = None,
        forward_kwargs: tp.Optional[tp.Sequence[str]] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an allocation function.

        Generates date points and allocates at those points.

        Similar to `PortfolioOptimizer.from_optimize_func`, but generates points using
        `vectorbtpro.base.wrapping.ArrayWrapper.get_index_points` and makes each point available
        as `index_point` in the context.

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.allocate_meta_nb`.

        Also, in contrast to `PortfolioOptimizer.from_optimize_func`, creates records of type
        `vectorbtpro.portfolio.pfopt.records.AllocPoints`.

        Usage:
            * Allocate uniformly:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np

            >>> data = vbt.YFData.fetch(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def uniform_allocate_func(n_cols):
            ...     return np.full(n_cols, 1 / n_cols)

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     uniform_allocate_func,
            ...     close.shape[1]
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2009-12-31 00:00:00+00:00  0.333333  0.333333  0.333333
            ```

            * Allocate randomly every first date of the year:

            ```pycon
            >>> def random_allocate_func(n_cols):
            ...     weights = np.random.uniform(size=n_cols)
            ...     return weights / weights.sum()

            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     every="AS-JAN"
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2011-01-03 00:00:00+00:00  0.152594  0.203128  0.644279
                        2012-01-03 00:00:00+00:00  0.707249  0.087783  0.204968
                        2013-01-02 00:00:00+00:00  0.327492  0.434563  0.237946
                        2014-01-02 00:00:00+00:00  0.237210  0.245863  0.516927
                        2015-01-02 00:00:00+00:00  0.339189  0.126962  0.533850
                        2016-01-04 00:00:00+00:00  0.140094  0.473617  0.386289
                        2017-01-03 00:00:00+00:00  0.476338  0.294500  0.229162
                        2018-01-02 00:00:00+00:00  0.195077  0.393477  0.411445
                        2019-01-02 00:00:00+00:00  0.297255  0.536558  0.166186
            ```

            * Specify index points manually:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60]
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2009-12-31 00:00:00+00:00  0.336081  0.313966  0.349953
                        2010-02-16 00:00:00+00:00  0.500909  0.282295  0.216796
                        2010-03-30 00:00:00+00:00  0.241952  0.556282  0.201767
            ```

            * Specify allocations manually:

            ```pycon
            >>> def manual_allocate_func(weights):
            ...     return weights

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     manual_allocate_func,
            ...     vbt.RepEval("weights[i]", context=dict(weights=[
            ...         [1, 0, 0],
            ...         [0, 1, 0],
            ...         [0, 0, 1]
            ...     ])),
            ...     index_points=[0, 30, 60]
            ... )
            >>> pf_opt.allocations
            symbol                                 MSFT  AMZN  AAPL
            alloc_group Date
            group       2009-12-31 00:00:00+00:00     1     0     0
                        2010-02-16 00:00:00+00:00     0     1     0
                        2010-03-30 00:00:00+00:00     0     0     1
            ```

            * Use Numba-compiled loop:

            ```pycon
            >>> from numba import njit

            >>> @njit
            ... def random_allocate_func_nb(i, idx, n_cols):
            ...     weights = np.random.uniform(0, 1, n_cols)
            ...     return weights / weights.sum()

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func_nb,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60],
            ...     jitted_loop=True
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2009-12-31 00:00:00+00:00  0.442137  0.233593  0.324270
                        2010-02-16 00:00:00+00:00  0.379956  0.309599  0.310445
                        2010-03-30 00:00:00+00:00  0.386918  0.228373  0.384709
            ```

            !!! hint
                There is no big reason of using the Numba-compiled loop, apart from when having
                to rebalance many thousands of times. Usually, using a regular Python loop
                and a Numba-compiled allocation function suffice.
        """
        if pbar_kwargs is None:
            pbar_kwargs = {}
        if isinstance(wrapper.columns, pd.MultiIndex):
            raise ValueError("Columns must represent only assets and cannot be multi-index")

        groupable_kwargs = {
            "allocate_func": allocate_func,
            "jitted_loop": jitted_loop,
            "every": every,
            "normalize_every": normalize_every,
            "start": start,
            "end": end,
            "on": on,
            "kind": kind,
            "jitted": jitted,
            "chunked": chunked,
            "index_points": index_points,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
            "forward_args": forward_args,
            "forward_kwargs": forward_kwargs,
            **kwargs,
        }
        if groups is None:
            groups = find_pfopt_groups(args, groupable_kwargs, sort_groups=sort_groups)
        if len(groups) == 0:
            groups = ["group"]
            single_group = True
        else:
            single_group = False

        alloc_points = []
        allocations = []
        if show_progress is None:
            show_progress = len(groups) > 1
            show_progress_none = True
        else:
            show_progress_none = False
        with get_pbar(total=len(groups), show_progress=show_progress, **pbar_kwargs) as pbar:
            for g, group in enumerate(groups):
                pbar.set_description(str(group))

                _args, _kwargs = select_pfopt_group_args(group, args, groupable_kwargs)
                _allocate_func = _kwargs.pop("allocate_func")
                _jitted_loop = _kwargs.pop("jitted_loop", False)
                _every = _kwargs.pop("every", None)
                _normalize_every = _kwargs.pop("normalize_every", False)
                _start = _kwargs.pop("start", None)
                _end = _kwargs.pop("end", None)
                _on = _kwargs.pop("on", None)
                _kind = _kwargs.pop("kind", None)
                _jitted = _kwargs.pop("jitted", None)
                _chunked = _kwargs.pop("chunked", None)
                _index_points = _kwargs.pop("index_points", None)
                _template_context = _kwargs.pop("template_context", None)
                _execute_kwargs = _kwargs.pop("execute_kwargs", None)
                _forward_args = _kwargs.pop("forward_args", None)
                _forward_kwargs = _kwargs.pop("forward_kwargs", None)

                _template_context = merge_dicts(
                    dict(
                        groups=groups,
                        group=group,
                        group_idx=g,
                        wrapper=wrapper,
                        allocate_func=_allocate_func,
                        jitted_loop=_jitted_loop,
                        every=_every,
                        normalize_every=_normalize_every,
                        start=_start,
                        end=_end,
                        on=_on,
                        jitted=_jitted,
                        chunked=_chunked,
                        index_points=_index_points,
                        args=_args,
                        kwargs=_kwargs,
                        execute_kwargs=_execute_kwargs,
                        forward_args=_forward_args,
                        forward_kwargs=_forward_kwargs,
                    ),
                    _template_context,
                )

                if _index_points is None:
                    get_index_points_kwargs = deep_substitute(
                        dict(
                            every=_every,
                            normalize_every=_normalize_every,
                            start=_start,
                            end=_end,
                            on=_on,
                            kind=_kind,
                        ),
                        _template_context,
                        sub_id="get_index_points_kwargs",
                        strict=True,
                    )
                    _index_points = wrapper.get_index_points(**get_index_points_kwargs)
                    _template_context = merge_dicts(
                        _template_context,
                        get_index_points_kwargs,
                        dict(index_points=_index_points),
                    )
                else:
                    _index_points = deep_substitute(
                        _index_points,
                        _template_context,
                        sub_id="index_points",
                        strict=True,
                    )
                    _index_points = to_1d_array(_index_points)

                if _forward_args is None:
                    _forward_args = []
                for k in _forward_args:
                    _args += (_template_context[k],)
                if _forward_kwargs is None:
                    _forward_kwargs = []
                for k in _forward_kwargs:
                    _kwargs[k] = _template_context[k]

                if jitted_loop:
                    _allocate_func = deep_substitute(
                        _allocate_func,
                        _template_context,
                        sub_id="allocate_func",
                        strict=True,
                    )
                    _args = deep_substitute(
                        _args,
                        _template_context,
                        sub_id="args",
                        strict=True,
                    )
                    _kwargs = deep_substitute(
                        _kwargs,
                        _template_context,
                        sub_id="kwargs",
                        strict=True,
                    )
                    func = jit_reg.resolve_option(nb.allocate_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    _allocations = func(len(wrapper.columns), _index_points, _allocate_func, *_args, **_kwargs)
                else:
                    funcs_args = []
                    for i in range(len(_index_points)):
                        __template_context = merge_dicts(dict(i=i, index_point=_index_points[i]), _template_context)
                        __allocate_func = deep_substitute(_allocate_func, _template_context, sub_id="optimize_func")
                        __args = deep_substitute(_args, __template_context, sub_id="args")
                        __kwargs = deep_substitute(_kwargs, __template_context, sub_id="kwargs")
                        funcs_args.append((__allocate_func, __args, __kwargs))

                    _execute_kwargs = merge_dicts(
                        dict(
                            show_progress=len(funcs_args) > 1 if show_progress_none else show_progress,
                            pbar_kwargs=pbar_kwargs,
                        ),
                        _execute_kwargs,
                    )
                    results = execute(funcs_args, **_execute_kwargs)
                    _allocations = pd.DataFrame(results, columns=wrapper.columns)
                    if isinstance(_allocations.columns, pd.RangeIndex):
                        _allocations = _allocations.values
                    else:
                        _allocations = _allocations[list(wrapper.columns)].values

                _alloc_points = np.empty(len(_allocations), alloc_point_dt)
                _alloc_points["id"] = np.arange(len(_allocations))
                _alloc_points["col"] = g
                _alloc_points["alloc_idx"] = _index_points
                alloc_points.append(_alloc_points)
                allocations.append(_allocations)

                pbar.update(1)

        if isinstance(groups, pd.Index):
            group_index = groups
        else:
            group_index = pd.Index(groups, name="alloc_group")
        if group_index.has_duplicates:
            raise ValueError("Groups cannot have duplicates")
        if group_index.name is None:
            raise ValueError("Group index must have a name")
        new_columns = combine_indexes((group_index, wrapper.columns))
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_points = AllocPoints(
            ArrayWrapper(
                index=wrapper.index,
                columns=group_index,
                ndim=1 if single_group else 2,
                freq=wrapper.freq,
                column_only_select=True,
            ),
            np.concatenate(alloc_points),
        )
        allocations = np.row_stack(allocations)
        return cls(new_wrapper, alloc_points, allocations)

    @classmethod
    def from_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocations: tp.ArrayLike,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from an array.

        Uses `PortfolioOptimizer.from_allocate_func`.

        If `allocations` is a NumPy array, uses `vectorbtpro.portfolio.pfopt.nb.pick_idx_allocate_func_nb`
        and a Numba-compiled loop. Otherwise, uses a regular Python function to pick each allocation
        (which can be a dict, Series, etc.).

        If `allocations` is a DataFrame, additionally uses its index as labels."""
        if isinstance(allocations, pd.DataFrame):
            kwargs = merge_dicts(
                dict(on=allocations.index, kind="labels"),
                kwargs,
            )
            allocations = allocations.values
        if isinstance(allocations, np.ndarray):

            def _resolve_allocations(index_points):
                if len(index_points) != len(allocations):
                    raise ValueError(f"Allocation array must have {len(index_points)} rows")
                return to_2d_array(allocations, expand_axis=0)

            return cls.from_allocate_func(
                wrapper,
                nb.pick_idx_allocate_func_nb,
                RepFunc(_resolve_allocations),
                jitted_loop=True,
                **kwargs,
            )

        def _pick_allocate_func(index_points, i):
            if len(index_points) != len(allocations):
                raise ValueError(f"Allocation array must have {len(index_points)} rows")
            return allocations[i]

        return cls.from_allocate_func(wrapper, _pick_allocate_func, Rep("index_points"), Rep("i"), **kwargs)

    @classmethod
    def from_filled_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        allocations: tp.AnyArray2d,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from an already filled array.

        Uses `PortfolioOptimizer.from_allocate_func`.

        Uses `vectorbtpro.portfolio.pfopt.nb.pick_point_allocate_func_nb` and a Numba-compiled loop.

        Extracts allocation points using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`."""
        if wrapper is None:
            if checks.is_frame(allocations):
                wrapper = ArrayWrapper.from_obj(allocations)
            else:
                raise TypeError("Must provide a wrapper if allocations is not a DataFrame")
        allocations = to_2d_array(allocations, expand_axis=0)
        if allocations.shape != wrapper.shape_2d:
            raise ValueError("Allocation array must have the same shape as wrapper")
        on = nb.get_alloc_points_nb(allocations, nonzero_only=nonzero_only, unique_only=unique_only)
        kwargs = merge_dicts(dict(on=on), kwargs)
        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            allocations,
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_uniform(cls: tp.Type[PortfolioOptimizerT], wrapper: ArrayWrapper, **kwargs) -> PortfolioOptimizerT:
        """Generate uniform allocations.

        Uses `PortfolioOptimizer.from_allocate_func`."""

        def _uniform_allocate_func():
            return np.full(wrapper.shape_2d[1], 1 / wrapper.shape_2d[1])

        return cls.from_allocate_func(wrapper, _uniform_allocate_func, **kwargs)

    @classmethod
    def from_random(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate random allocations.

        Uses `PortfolioOptimizer.from_allocate_func`.

        Uses `vectorbtpro.portfolio.pfopt.nb.random_allocate_func_nb` and a Numba-compiled loop."""
        if seed is not None:
            set_seed_nb(seed)
        return cls.from_allocate_func(
            wrapper,
            nb.random_allocate_func_nb,
            wrapper.shape_2d[1],
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_universal_algo(
        cls: tp.Type[PortfolioOptimizerT],
        algo: tp.Union[str, tp.Type[AlgoT], AlgoT, AlgoResultT],
        S: tp.Optional[tp.AnyArray2d] = None,
        n_jobs: int = 1,
        log_progress: bool = False,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations using [Universal Portfolios](https://github.com/Marigold/universal-portfolios).

        `S` can be any price, while `algo` must be either an attribute of the package, subclass of
        `universal.algo.Algo`, instance of `universal.algo.Algo`, or instance of `universal.result.AlgoResult`.

        Extracts allocation points using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`."""
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("universal")
        from universal.algo import Algo
        from universal.result import AlgoResult

        if wrapper is None:
            if S is None or not checks.is_frame(S):
                raise TypeError("Must provide a wrapper if allocations is not a DataFrame")
            else:
                wrapper = ArrayWrapper.from_obj(S)

        group_weights = {}

        if isinstance(algo, str):
            import universal.algos

            algo = getattr(universal.algos, algo)
        if isinstance(algo, type) and issubclass(algo, Algo):
            algo_arg_names = get_func_arg_names(algo)
            algo_kwargs = {}
            for k in algo_arg_names:
                if k in kwargs:
                    algo_kwargs[k] = kwargs.pop(k)
            algo = algo(**algo_kwargs)

        def _resolve_algo_result(group, kwargs, _algo=algo):
            _kwargs = dict(kwargs)
            for k in list(kwargs.keys()):
                kwargs.pop(k)  # otherwise gets passed to Numba function
            nonlocal group_weights
            if group in group_weights:
                return group_weights[group]
            if isinstance(_algo, Algo):
                if S is None:
                    raise ValueError("Must provide S")
                _algo = _algo.run(S, n_jobs=n_jobs, log_progress=log_progress)
            if isinstance(_algo, AlgoResult):
                group_weights[group] = _algo.weights[wrapper.columns].values
                return group_weights[group]
            raise TypeError(f"Algo {_algo} not supported")

        def _resolve_index_points(group, kwargs):
            weights = _resolve_algo_result(group, kwargs)
            return nb.get_alloc_points_nb(weights, nonzero_only=nonzero_only, unique_only=unique_only)

        allocations = RepFunc(_resolve_algo_result)
        on = RepFunc(_resolve_index_points)
        kwargs = merge_dicts(dict(on=on), kwargs)
        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            allocations,
            jitted_loop=True,
            **kwargs,
        )

    # ############# Properties ############# #

    @property
    def alloc_records(self) -> tp.Union[AllocRanges, AllocPoints]:
        """Allocation ranges of type `vectorbtpro.portfolio.pfopt.records.AllocRanges`
        or points of type `vectorbtpro.portfolio.pfopt.records.AllocPoints`."""
        return self._alloc_records

    def get_allocations(self, squeeze_groups: bool = False) -> tp.Frame:
        """Get a DataFrame with allocation groups concatenated along the index axis."""
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            closed_mask = self.alloc_records.get_field_arr("status") == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            index = self.wrapper.index[idx_arr]
        else:
            index = stack_indexes((self.alloc_records.wrapper.columns[group_arr], self.wrapper.index[idx_arr]))
        columns = self.wrapper.columns.unique(level=1)
        return pd.DataFrame(allocations, index=index, columns=columns)

    @property
    def allocations(self) -> tp.Frame:
        """Calls `PortfolioOptimizer.get_allocations` with default arguments."""
        return self.get_allocations()

    def fill_allocations(
        self,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        wrap_kwargs: tp.KwargsLike = None,
        squeeze_groups: bool = False,
    ) -> tp.Frame:
        """Fill an empty DataFrame with allocations.

        Set `dropna` to 'all' to remove all NaN rows, or to 'head' to remove any rows coming before
        the first allocation."""
        if wrap_kwargs is None:
            wrap_kwargs = {}
        out = self.wrapper.fill(fill_value, group_by=False, **wrap_kwargs)
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            closed_mask = self.alloc_records.get_field_arr("status") == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        for g in range(len(self.alloc_records.wrapper.columns)):
            group_mask = group_arr == g
            index = self.wrapper.index[idx_arr[group_mask]]
            column_mask = self.wrapper.columns.get_level_values(level=0) == self.alloc_records.wrapper.columns[g]
            columns = self.wrapper.columns[column_mask]
            out.loc[index, columns] = allocations[group_mask]
        if dropna is not None:
            if dropna.lower() == "all":
                out = out.dropna(how="all")
            elif dropna.lower() == "head":
                out = out.iloc[idx_arr.min() :]
            else:
                raise ValueError(f"Invalid option dropna='{dropna}'")
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            out = out.droplevel(level=0, axis=1)
        return out

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `PortfolioOptimizer.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.pfopt`."""
        from vectorbtpro._settings import settings

        pfopt_stats_cfg = settings["pfopt"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), pfopt_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(title="Start", calc_func=lambda self: self.wrapper.index[0], agg_func=None, tags="wrapper"),
            end=dict(title="End", calc_func=lambda self: self.wrapper.index[-1], agg_func=None, tags="wrapper"),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            coverage=dict(
                title="Coverage",
                calc_func="alloc_records.coverage",
                overlapping=False,
                normalize=False,
                apply_to_timedelta=True,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="alloc_records.coverage",
                overlapping=True,
                normalize=False,
                apply_to_timedelta=True,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            total_records=dict(title="Total Records", calc_func="alloc_records.count", tags="alloc_records"),
            mean_allocation=dict(
                title="Mean Allocation",
                calc_func=lambda allocations: to_dict(
                    allocations.groupby(level=0).mean().transpose(),
                    orient="index_series",
                ),
                resolve_allocations=True,
                tags="allocations",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        dropna: tp.Optional[str] = "head",
        plot_rb_dates: tp.Optional[bool] = False,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot allocations.

        Args:
            column (str): Name of the allocation group to plot.
            dropna (int): See `PortfolioOptimizer.fill_allocations`.
            plot_rb_dates (bool): Whether to plot rebalancing dates.

                Defaults to True if there are no more than 20 rebalancing dates.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` for rebalancing dates.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            * Continuing with the examples under `PortfolioOptimizer.from_optimize_func`:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> pf_opt = vbt.PortfolioOptimizer.from_random(
            ...     vbt.ArrayWrapper(
            ...         index=pd.date_range("2020-01-01", "2021-01-01"),
            ...         columns=["MSFT", "AMZN", "AAPL"],
            ...         ndim=2
            ...     ),
            ...     every="MS",
            ...     seed=40
            ... )
            >>> pf_opt.plot()
            ```

            ![](/assets/images/pfopt_plot.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.express as px

        self_group = self.select_col(column=column)
        filled_alloc_df = self_group.fill_allocations(dropna=dropna).ffill()
        columns = self_group.wrapper.columns.unique(level=1)
        if len(columns) <= len(px.colors.qualitative.D3):
            colors = px.colors.qualitative.D3
        else:
            colors = px.colors.qualitative.Alphabet

        fig = filled_alloc_df.droplevel(level=0, axis=1).vbt.plot(
            trace_kwargs=[
                merge_dicts(
                    dict(
                        stackgroup="one",
                        groupnorm="percent",
                        line=dict(width=0),
                        fillcolor=adjust_opacity(colors[c % len(colors)], 0.8),
                    ),
                    resolve_dict(trace_kwargs, i=c),
                )
                for c in range(len(columns))
            ],
            add_trace_kwargs=add_trace_kwargs,
            use_gl=False,
            fig=fig,
            **layout_kwargs,
        )

        if plot_rb_dates is None or (isinstance(plot_rb_dates, bool) and plot_rb_dates):
            rb_dates = self_group.allocations.index.get_level_values(level=1)
            if plot_rb_dates is None:
                plot_rb_dates = len(rb_dates) <= 20
            if plot_rb_dates:
                add_shape_kwargs = merge_dicts(
                    dict(
                        type="line",
                        line=dict(
                            color=fig.layout.template.layout.plot_bgcolor,
                            dash="dot",
                            width=1,
                        ),
                        xref="x",
                        yref="paper",
                        y0=0,
                        y1=1,
                    ),
                    add_shape_kwargs,
                )
                for rb_date in rb_dates:
                    fig.add_shape(x0=rb_date, x1=rb_date, **add_shape_kwargs)
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `PortfolioOptimizer.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.pfopt`."""
        from vectorbtpro._settings import settings

        pfopt_plots_cfg = settings["pfopt"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), pfopt_plots_cfg)

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            alloc_ranges=dict(
                title="Allocation Ranges",
                plot_func="alloc_records.plot",
                check_alloc_ranges=True,
                tags="alloc_ranges",
            ),
            plot=dict(
                title="Allocations",
                plot_func="plot",
                tags="allocations",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


PortfolioOptimizer.override_metrics_doc(__pdoc__)
PortfolioOptimizer.override_subplots_doc(__pdoc__)
