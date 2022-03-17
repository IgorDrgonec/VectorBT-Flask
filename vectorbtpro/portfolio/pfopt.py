# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Functions for portfolio optimization."""

from vectorbtpro.utils.opt_packages import assert_can_import

assert_can_import("pypfopt")

import inspect

from pypfopt.base_optimizer import BaseOptimizer

from vectorbtpro import _typing as tp
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.parsing import get_func_arg_names


class pfopt_func_dict(dict):
    """Dict that contains optimization functions as keys.

    Keys can be functions themselves, their names, or `_default` for the default value."""

    pass


def select_pfopt_func_kwargs(pfopt_func: tp.Callable, kwargs: tp.DictLike) -> dict:
    """Select keyword arguments belonging to `pfopt_func`."""
    if isinstance(kwargs, pfopt_func_dict):
        kwargs = kwargs[pfopt_func]
    if kwargs is None:
        kwargs = {}
    _kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, pfopt_func_dict):
            if pfopt_func in v or pfopt_func.__name__ in v:
                _kwargs[k] = v[pfopt_func]
            elif "_default" in v:
                _kwargs[k] = v["_default"]
        else:
            _kwargs[k] = v
    return _kwargs


def resolve_pfopt_func_kwargs(
    pfopt_func: tp.Callable,
    cache: tp.KwargsLike = None,
    var_kwarg_names: tp.Optional[tp.Iterable[str]] = None,
    **kwargs,
) -> tp.Kwargs:
    """Resolve keyword arguments passed to any optimization function with the layout of PyPortfolioOpt.

    Parses the signature of `pfopt_func`, and for each accepted argument, looks for an argument
    with the same name in `kwargs`. If not found, tries to resolve that argument using other arguments
    or by calling other optimization functions.

    Argument `frequency` gets resolved with (global) `freq` and `year_freq` using
    `vectorbtpro.returns.accessors.ReturnsAccessor.get_ann_factor`.

    Any argument in `kwargs` can be wrapped using `pfopt_func_dict` to define the argument
    per function rather than globally.

    !!! note
        When providing custom functions, make sure that the arguments they accept are visible
        in the signature (that is, no variable arguments) and have the same naming as in PyPortfolioOpt.

        Functions `market_implied_prior_returns` and `BlackLittermanModel.bl_weights` take `risk_aversion`,
        which is different from arguments with the same name in other functions. To set it, pass `delta`."""
    signature = inspect.signature(pfopt_func)
    kwargs = select_pfopt_func_kwargs(pfopt_func, kwargs)
    if cache is None:
        cache = {}
    arg_names = get_func_arg_names(pfopt_func)
    if len(arg_names) == 0:
        return {}

    pass_kwargs = dict()

    def _process_arg(arg_name, arg_value):
        orig_arg_name = arg_name
        if pfopt_func.__name__ in ("market_implied_prior_returns", "bl_weights"):
            if arg_name == "risk_aversion":
                # In some methods, risk_aversion is expected as array and means delta
                arg_name = "delta"

        prices = None
        returns = None
        if "prices" in cache:
            prices = cache["prices"]
        elif "prices" in kwargs:
            if kwargs.get("returns_data", False):
                returns = kwargs["prices"]
            else:
                prices = kwargs["prices"]
        if "returns" in cache:
            returns = cache["returns"]
        elif "returns" in kwargs:
            returns = kwargs["returns"]
        elif "prices" in kwargs and kwargs.get("returns_data", False):
            returns = kwargs["prices"]

        def _prices_from_returns(_returns=returns):
            from pypfopt.expected_returns import prices_from_returns

            cache["prices"] = prices_from_returns(_returns, kwargs.get("log_returns", False))
            return cache["prices"]

        def _returns_from_prices(_prices=prices):
            from pypfopt.expected_returns import returns_from_prices

            cache["returns"] = returns_from_prices(_prices, kwargs.get("log_returns", False))
            return cache["returns"]

        if arg_name == "expected_returns":
            if "expected_returns" not in cache:
                cache["expected_returns"] = resolve_expected_returns(cache=cache, **kwargs)
            pass_kwargs[orig_arg_name] = cache["expected_returns"]
        elif arg_name == "cov_matrix":
            if "cov_matrix" not in cache:
                cache["cov_matrix"] = resolve_cov_matrix(cache=cache, **kwargs)
            pass_kwargs[orig_arg_name] = cache["cov_matrix"]
        elif arg_name == "optimizer":
            if "optimizer" not in cache:
                cache["optimizer"] = resolve_optimizer(cache=cache, **kwargs)
            pass_kwargs[orig_arg_name] = cache["optimizer"]

        if orig_arg_name not in pass_kwargs:
            if arg_name in kwargs:
                if arg_name == "market_prices":
                    if pfopt_func.__name__ != "market_implied_risk_aversion" and checks.is_series(kwargs[arg_name]):
                        pass_kwargs[orig_arg_name] = kwargs[arg_name].to_frame().copy(deep=False)
                    else:
                        pass_kwargs[orig_arg_name] = kwargs[arg_name].copy(deep=False)
                else:
                    pass_kwargs[orig_arg_name] = kwargs[arg_name]
            else:
                if arg_name == "frequency":
                    ann_factor = ReturnsAccessor.get_ann_factor(kwargs.get("year_freq", None), kwargs.get("freq", None))
                    if ann_factor is not None:
                        pass_kwargs[orig_arg_name] = ann_factor
                elif arg_name == "prices":
                    if "returns_data" in arg_names:
                        if "returns_data" in kwargs:
                            if kwargs.get("returns_data", False):
                                if returns is not None:
                                    pass_kwargs[orig_arg_name] = returns
                                elif prices is not None:
                                    pass_kwargs[orig_arg_name] = _returns_from_prices()
                            else:
                                if prices is not None:
                                    pass_kwargs[orig_arg_name] = prices
                                elif returns is not None:
                                    pass_kwargs[orig_arg_name] = _prices_from_returns()
                        else:
                            if prices is not None:
                                pass_kwargs[orig_arg_name] = prices
                                pass_kwargs["returns_data"] = False
                            elif returns is not None:
                                pass_kwargs[orig_arg_name] = returns
                                pass_kwargs["returns_data"] = True
                    else:
                        if prices is not None:
                            pass_kwargs[orig_arg_name] = prices
                        elif returns is not None:
                            pass_kwargs[orig_arg_name] = _prices_from_returns()
                elif arg_name == "returns":
                    if returns is not None:
                        pass_kwargs[orig_arg_name] = returns
                    elif prices is not None:
                        pass_kwargs[orig_arg_name] = _returns_from_prices()
                elif arg_name == "latest_prices":
                    from pypfopt.discrete_allocation import get_latest_prices

                    if prices is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(prices)
                    elif returns is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(_prices_from_returns())
                elif arg_name == "delta":
                    if "delta" not in cache:
                        from pypfopt.black_litterman import market_implied_risk_aversion

                        cache["delta"] = resolve_pfopt_func_call(market_implied_risk_aversion, cache=cache, **kwargs)
                    pass_kwargs[orig_arg_name] = cache["delta"]
                elif arg_name == "pi":
                    if "pi" not in cache:
                        from pypfopt.black_litterman import market_implied_prior_returns

                        cache["pi"] = resolve_pfopt_func_call(market_implied_prior_returns, cache=cache, **kwargs)
                    pass_kwargs[orig_arg_name] = cache["pi"]

        if orig_arg_name not in pass_kwargs:
            if arg_value.default != inspect.Parameter.empty:
                pass_kwargs[orig_arg_name] = arg_value.default

    for arg_name, arg_value in signature.parameters.items():
        if arg_value.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"Variable positional arguments in {pfopt_func} cannot be parsed")
        elif arg_value.kind == inspect.Parameter.VAR_KEYWORD:
            if var_kwarg_names is None:
                var_kwarg_names = []
            for var_arg_name in var_kwarg_names:
                _process_arg(var_arg_name, arg_value)
        else:
            _process_arg(arg_name, arg_value)

    return pass_kwargs


def resolve_pfopt_func_call(pfopt_func: tp.Callable, **kwargs) -> tp.Any:
    """Resolve arguments using `resolve_pfopt_func_kwargs` and call the function with that arguments."""
    return pfopt_func(**resolve_pfopt_func_kwargs(pfopt_func, **kwargs))


def resolve_expected_returns(
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

    Any function is resolved using `resolve_pfopt_func_call`."""
    if isinstance(expected_returns, str):
        if expected_returns.lower() == "mean_historical_return":
            from pypfopt.expected_returns import mean_historical_return

            return resolve_pfopt_func_call(mean_historical_return, **kwargs)
        if expected_returns.lower() == "ema_historical_return":
            from pypfopt.expected_returns import ema_historical_return

            return resolve_pfopt_func_call(ema_historical_return, **kwargs)
        if expected_returns.lower() == "capm_return":
            from pypfopt.expected_returns import capm_return

            return resolve_pfopt_func_call(capm_return, **kwargs)
        if expected_returns.lower() == "bl_returns":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            ).bl_returns()
        import pypfopt.expected_returns

        if hasattr(pypfopt.expected_returns, expected_returns):
            return resolve_pfopt_func_call(getattr(pypfopt.expected_returns, expected_returns), **kwargs)
        raise NotImplementedError("Return model '{}' not supported".format(expected_returns))
    if callable(expected_returns):
        return resolve_pfopt_func_call(expected_returns, **kwargs)
    return expected_returns


def resolve_cov_matrix(
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

    Any function is resolved using `resolve_pfopt_func_call`."""
    if isinstance(cov_matrix, str):
        if cov_matrix.lower() == "sample_cov":
            from pypfopt.risk_models import sample_cov

            return resolve_pfopt_func_call(sample_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "semicovariance" or cov_matrix.lower() == "semivariance":
            from pypfopt.risk_models import semicovariance

            return resolve_pfopt_func_call(semicovariance, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "exp_cov":
            from pypfopt.risk_models import exp_cov

            return resolve_pfopt_func_call(exp_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "ledoit_wolf" or cov_matrix.lower() == "ledoit_wolf_constant_variance":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf()
        if cov_matrix.lower() == "ledoit_wolf_single_factor":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(shrinkage_target="single_factor")
        if cov_matrix.lower() == "ledoit_wolf_constant_correlation":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="constant_correlation"
            )
        if cov_matrix.lower() == "oracle_approximating":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pfopt_func_call(CovarianceShrinkage, **kwargs).oracle_approximating()
        import pypfopt.risk_models

        if hasattr(pypfopt.risk_models, cov_matrix):
            return resolve_pfopt_func_call(getattr(pypfopt.risk_models, cov_matrix), **kwargs)
        raise NotImplementedError("Risk model '{}' not supported".format(cov_matrix))
    if callable(cov_matrix):
        return resolve_pfopt_func_call(cov_matrix, **kwargs)
    return cov_matrix


BaseOptimizerT = tp.TypeVar("BaseOptimizerT", bound=BaseOptimizer)


def resolve_optimizer(
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

    Any function is resolved using `resolve_pfopt_func_call`."""
    if isinstance(optimizer, str):
        if optimizer.lower() == "efficient_frontier":
            from pypfopt.efficient_frontier import EfficientFrontier

            return resolve_pfopt_func_call(EfficientFrontier, **kwargs)
        if optimizer.lower() == "efficient_cdar":
            from pypfopt.efficient_frontier import EfficientCDaR

            return resolve_pfopt_func_call(EfficientCDaR, **kwargs)
        if optimizer.lower() == "efficient_cvar":
            from pypfopt.efficient_frontier import EfficientCVaR

            return resolve_pfopt_func_call(EfficientCVaR, **kwargs)
        if optimizer.lower() == "efficient_semivariance":
            from pypfopt.efficient_frontier import EfficientSemivariance

            return resolve_pfopt_func_call(EfficientSemivariance, **kwargs)
        if optimizer.lower() == "black_litterman" or optimizer.lower() == "bl":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            )
        if optimizer.lower() == "hierarchical_portfolio" or optimizer.lower() == "hrpopt" or optimizer.lower() == "hrp":
            from pypfopt.hierarchical_portfolio import HRPOpt

            return resolve_pfopt_func_call(HRPOpt, **kwargs)
        if optimizer.lower() == "cla":
            from pypfopt.cla import CLA

            return resolve_pfopt_func_call(CLA, **kwargs)
        import pypfopt

        if hasattr(pypfopt, optimizer):
            return resolve_pfopt_func_call(getattr(pypfopt, optimizer), **kwargs)
        raise NotImplementedError("Optimizer '{}' not supported".format(optimizer))
    if isinstance(optimizer, type) and issubclass(optimizer, BaseOptimizer):
        return resolve_pfopt_func_call(optimizer, **kwargs)
    if isinstance(optimizer, BaseOptimizer):
        return optimizer
    raise NotImplementedError("Optimizer {} not supported".format(optimizer))


def get_allocation(
    target: tp.Union[tp.Callable, str] = "max_sharpe",
    target_is_convex: bool = True,
    weights_sum_to_one: bool = True,
    target_constraints: tp.Optional[tp.List[tp.Kwargs]] = None,
    target_solver: str = "SLSQP",
    target_initial_guess: tp.Optional[tp.Array] = None,
    objectives: tp.Optional[tp.MaybeIterable[tp.Union[tp.Callable, str]]] = None,
    constraints: tp.Optional[tp.MaybeIterable[tp.Callable]] = None,
    sector_mapper: tp.Optional[dict] = None,
    sector_lower: tp.Optional[dict] = None,
    sector_upper: tp.Optional[dict] = None,
    discrete_allocation: bool = False,
    allocation_method: str = "lp_portfolio",
    **kwargs,
) -> tp.Dict[str, float]:
    """Get allocation.

    First, it resolves the optimizer using `resolve_optimizer`. Depending upon which arguments it takes,
    it may further resolve expected returns, covariance matrix, etc. Then, it adds objectives and constraints
    to the optimizer instance, calls the target metric, extracts the weights, and finally, converts
    the weights to an integer allocation (if requested).

    To specify the optimizer, use `optimizer` (see `resolve_optimizer`).
    To specify the expected returns, use `expected_returns` (see `resolve_expected_returns`).
    To specify the covariance matrix, use `cov_matrix` (see `resolve_cov_matrix`).
    All other keyword arguments in `**kwargs` are used by `resolve_pfopt_func_call`.

    Each objective can be a function, an attribute of `pypfopt.objective_functions`, or an iterable of such.

    Each constraint can be a function or an interable of such.

    The target can be an attribute of the optimizer, or a stand-alone function.
    If `target_is_convex` is True, the function is added as a convex function.
    Otherwise, the function is added as a non-convex function. The keyword arguments
    `weights_sum_to_one` and those starting with `target` are passed
    `pypfopt.base_optimizer.BaseConvexOptimizer.convex_objective`
    and `pypfopt.base_optimizer.BaseConvexOptimizer.nonconvex_objective` respectively.

    If `discrete_allocation` is True, resolves `pypfopt.discrete_allocation.DiscreteAllocation`
    and calls `allocation_method` as an attribute of the allocation object.

    Any function is resolved using `resolve_pfopt_func_call`.

    Usage:
        * Using mean historical returns, Ledoit-Wolf covariance matrix with constant variance,
        and efficient frontier:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> from vectorbtpro.portfolio.pfopt import get_allocation

        >>> data = vbt.YFData.fetch(["MSFT", "AMZN", "KO", "MA"])
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> get_allocation(prices=data.get("Close"))
        OrderedDict([('MSFT', 0.13082),
                     ('AMZN', 0.10451),
                     ('KO', 0.02513),
                     ('MA', 0.73954)])
        ```

        * EMA historical returns and sample covariance:

        ```pycon
        >>> get_allocation(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     cov_matrix="sample_cov"
        ... )
        OrderedDict([('MSFT', 0.46812), ('AMZN', 0.0), ('KO', 0.53188), ('MA', 0.0)])
        ```

        * EMA historical returns, efficient Conditional Value at Risk, and other parameters automatically
        passed to their respective functions. Optimized towards lowest CVaR:

        ```pycon
        >>> get_allocation(
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
        >>> get_allocation(
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
        >>> get_allocation(
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

        >>> get_allocation(
        ...     prices=data.get("Close"),
        ...     target=logarithmic_barrier_objective
        ... )
        OrderedDict([('MSFT', 0.24599),
                     ('AMZN', 0.23072),
                     ('KO', 0.25823),
                     ('MA', 0.26507)])
        ```
    """
    cache = {}
    optimizer = kwargs["optimizer"] = resolve_optimizer(cache=cache, **kwargs)
    if objectives is not None:
        if not checks.is_iterable(objectives) or isinstance(objectives, str):
            objectives = [objectives]
        for objective in objectives:
            if isinstance(objective, str):
                import pypfopt.objective_functions

                objective = getattr(pypfopt.objective_functions, objective)
            objective_kwargs = resolve_pfopt_func_kwargs(objective, cache=cache, **kwargs)
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
    if isinstance(target, str):
        resolve_pfopt_func_call(getattr(optimizer, target), cache=cache, **kwargs)
    else:
        if target_is_convex:
            optimizer.convex_objective(
                target,
                weights_sum_to_one=weights_sum_to_one,
                **resolve_pfopt_func_kwargs(target, cache=cache, **kwargs),
            )
        else:
            optimizer.nonconvex_objective(
                target,
                objective_args=tuple(resolve_pfopt_func_kwargs(target, cache=cache, **kwargs).values()),
                weights_sum_to_one=weights_sum_to_one,
                constraints=target_constraints,
                solver=target_solver,
                initial_guess=target_initial_guess,
            )
    weights = kwargs["weights"] = resolve_pfopt_func_call(optimizer.clean_weights, cache=cache, **kwargs)
    if discrete_allocation:
        from pypfopt.discrete_allocation import DiscreteAllocation

        allocator = resolve_pfopt_func_call(DiscreteAllocation, cache=cache, **kwargs)
        return resolve_pfopt_func_call(getattr(allocator, allocation_method), cache=cache, **kwargs)[0]
    return weights
