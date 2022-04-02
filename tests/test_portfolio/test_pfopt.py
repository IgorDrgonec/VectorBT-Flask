import os
import pytest

import numpy as np
import pandas as pd
from numba import njit

import vectorbtpro as vbt
import vectorbtpro.portfolio.pfopt.base as pfopt
from vectorbtpro.portfolio.enums import alloc_range_dt, alloc_point_dt
from tests.utils import assert_records_close

pypfopt_available = True
try:
    import pypfopt
except:
    pypfopt_available = False
universal_available = True
try:
    import universal
except:
    universal_available = False

# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# PyPortfolioOpt ############# #

prices = pd.DataFrame(
    {
        "XOM": [
            54.068794,
            54.279907,
            54.749043,
            54.577045,
            54.358093,
        ],
        "RRC": [
            51.300568,
            51.993038,
            51.690697,
            51.593170,
            52.597733,
        ],
        "BBY": [
            32.524055,
            33.349487,
            33.090542,
            33.616547,
            32.297466,
        ],
        "MA": [
            22.062426,
            21.997149,
            22.081820,
            21.937523,
            21.945297,
        ],
        "PFE": [
            13.940202,
            13.741367,
            13.697187,
            13.645634,
            13.756095,
        ],
    },
    index=pd.date_range("2020-01-01", "2020-01-05"),
)
market_caps = prices.prod(axis=0)
returns = prices.pct_change().dropna(how="all")
viewdict = {"XOM": 0.20, "RRC": -0.30, "BBY": 0, "MA": -0.2, "PFE": 0.15}


class TestPyPortfolioOpt:
    def test_resolve_pypfopt_expected_returns(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", prices=prices),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", prices=prices, returns_data=False),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", prices=returns, returns_data=True),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", returns=returns),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", prices=prices, returns=returns),
                pypfopt.expected_returns.mean_historical_return(prices, returns_data=False),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "mean_historical_return",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                ),
                pypfopt.expected_returns.mean_historical_return(prices, frequency=365, compounding=False),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "ema_historical_return",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                    span=350,
                ),
                pypfopt.expected_returns.ema_historical_return(prices, frequency=365, compounding=False, span=350),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "capm_return",
                    prices=prices,
                    market_prices=prices.iloc[:, 0],
                    risk_free_rate=0.03,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                    span=350,
                ),
                pypfopt.expected_returns.capm_return(
                    prices,
                    frequency=365,
                    market_prices=prices.iloc[:, 0].to_frame(),
                    risk_free_rate=0.03,
                    compounding=False,
                ),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "bl_returns",
                    prices=prices,
                    absolute_views=viewdict,
                    pi=None,
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    lambda prices, absolute_views: pypfopt.black_litterman.BlackLittermanModel(
                        pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                        absolute_views=viewdict,
                    ).bl_returns(),
                    prices=prices,
                    absolute_views=viewdict,
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    pypfopt.black_litterman.BlackLittermanModel(
                        pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                        absolute_views=viewdict,
                    ).bl_returns(),
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )

    def test_resolve_pypfopt_cov_matrix(self):
        if pypfopt_available:
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "sample_cov",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    fix_method="diag",
                ),
                pypfopt.risk_models.sample_cov(
                    prices,
                    frequency=365,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "exp_cov",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    benchmark=0.1,
                    fix_method="diag",
                ),
                pypfopt.risk_models.exp_cov(
                    prices,
                    frequency=365,
                    benchmark=0.1,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "exp_cov",
                    returns=returns,
                    freq="1d",
                    year_freq="365 days",
                    span=200,
                    fix_method="diag",
                ),
                pypfopt.risk_models.exp_cov(
                    returns,
                    returns_data=True,
                    frequency=365,
                    span=200,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf_single_factor",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).ledoit_wolf(shrinkage_target="single_factor"),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf_constant_correlation",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).ledoit_wolf(shrinkage_target="constant_correlation"),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "oracle_approximating",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    lambda prices, frequency: pypfopt.risk_models.CovarianceShrinkage(
                        prices,
                        frequency=frequency,
                    ).oracle_approximating(),
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    pypfopt.risk_models.CovarianceShrinkage(
                        prices,
                        frequency=365,
                    ).oracle_approximating(),
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )

    def test_resolve_pypfopt_optimizer(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            weights1 = pfopt.resolve_pypfopt_optimizer("efficient_frontier", prices=prices).min_volatility()
            weights2 = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_frontier",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_cdar",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cdar()
            weights2 = pypfopt.efficient_frontier.EfficientCDaR(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cdar()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_cvar",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cvar()
            weights2 = pypfopt.efficient_frontier.EfficientCVaR(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cvar()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_semivariance",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                benchmark=0.1,
                weight_bounds=(-1, 1),
            ).min_semivariance()
            weights2 = pypfopt.efficient_frontier.EfficientSemivariance(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                benchmark=0.1,
                weight_bounds=(-1, 1),
            ).min_semivariance()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "black_litterman",
                prices=prices,
                pi=None,
                absolute_views=viewdict,
            ).bl_weights()
            weights2 = pypfopt.black_litterman.BlackLittermanModel(
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                absolute_views=viewdict,
            ).bl_weights()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "black_litterman",
                prices=prices,
                market_caps=market_caps,
                market_prices=prices.iloc[:, 0],
                absolute_views=viewdict,
                freq="1d",
                year_freq="365 days",
                risk_free_rate=0.03,
            ).bl_weights()
            weights2 = pypfopt.black_litterman.BlackLittermanModel(
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                pi=pypfopt.black_litterman.market_implied_prior_returns(
                    market_caps,
                    pypfopt.black_litterman.market_implied_risk_aversion(
                        prices.iloc[:, 0],
                        frequency=365,
                        risk_free_rate=0.03,
                    ),
                    pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                    risk_free_rate=0.03,
                ),
                absolute_views=viewdict,
            ).bl_weights()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "hrpopt",
                prices=prices,
                freq="1d",
                year_freq="365 days",
            ).optimize()
            weights2 = pypfopt.hierarchical_portfolio.HRPOpt(
                returns,
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
            ).optimize()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "cla",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2

            from pypfopt.cla import CLA

            class CustomCLA(CLA):
                def __init__(self, prices, frequency=252, weight_bounds=(0, 1)):
                    expected_returns = pypfopt.expected_returns.mean_historical_return(
                        prices=prices, frequency=frequency
                    )
                    cov_matrix = pypfopt.risk_models.CovarianceShrinkage(
                        prices=prices, frequency=frequency
                    ).ledoit_wolf()
                    CLA.__init__(self, expected_returns, cov_matrix, weight_bounds=weight_bounds)

            weights1 = pfopt.resolve_pypfopt_optimizer(
                CustomCLA,
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                pypfopt.cla.CLA(
                    pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                    pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                    weight_bounds=(-1, 1),
                )
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2

    def test_pypfopt_optimize(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.max_sharpe()
            assert pfopt.pypfopt_optimize(prices=prices) == ef.clean_weights()
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.ema_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(shrinkage_target="single_factor"),
            )
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    expected_returns="ema_historical_return",
                    cov_matrix="ledoit_wolf_single_factor",
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientCVaR(
                pypfopt.expected_returns.ema_historical_return(prices=prices),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            )
            ef.min_cvar()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target="min_cvar",
                    expected_returns="ema_historical_return",
                    optimizer="efficient_cvar",
                    beta=0.9,
                    weight_bounds=(-1, 1),
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            from pypfopt import objective_functions

            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives="L2_reg",
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=["L2_reg"],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=objective_functions.L2_reg,
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=[objective_functions.L2_reg],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

            da = DiscreteAllocation(ef.clean_weights(), get_latest_prices(prices), total_portfolio_value=20000)
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives="L2_reg",
                    gamma=0.1,
                    discrete_allocation=True,
                    total_portfolio_value=20000,
                )
                == da.lp_portfolio()[0]
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_constraint(lambda w: w[0] >= 0.2)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=lambda w: w[0] >= 0.2,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=[lambda w: w[0] >= 0.2],
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_constraint(lambda w: w[0] >= 0.2)
            ef.add_constraint(lambda w: w >= 0.1)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=[
                        lambda w: w[0] >= 0.2,
                        lambda w: w >= 0.1,
                    ],
                )
                == ef.clean_weights()
            )

            sector_mapper = {
                "XOM": "tech",
                "RRC": "tech",
                "BBY": "Oil/Gas",
                "MA": "Oil/Gas",
                "PFE": "Financials",
            }
            sector_lower = {"tech": 0.1}
            sector_upper = {"tech": 0.4, "Oil/Gas": 0.1}
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    sector_mapper=sector_mapper,
                    sector_lower=sector_lower,
                    sector_upper=sector_upper,
                )
                == ef.clean_weights()
            )

            import cvxpy as cp

            def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
                log_sum = cp.sum(cp.log(w))
                var = cp.quad_form(w, cov_matrix)
                return var - k * log_sum

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                weight_bounds=(0.01, 0.2),
            )
            ef.convex_objective(
                logarithmic_barrier_objective,
                cov_matrix=pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                k=0.001,
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target=logarithmic_barrier_objective,
                    k=0.001,
                    weight_bounds=(0.01, 0.2),
                )
                == ef.clean_weights()
            )

            def deviation_risk_parity(w, cov_matrix):
                diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
                return (diff ** 2).sum().sum()

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                weight_bounds=(0.01, 0.12),
            )
            ef.nonconvex_objective(
                deviation_risk_parity,
                objective_args=(pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),),
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target=deviation_risk_parity,
                    target_is_convex=False,
                    weight_bounds=(0.01, 0.12),
                )
                == ef.clean_weights()
            )


# ############# PortfolioOptimizer ############# #

class TestPortfolioOptimizer:
    def test_find_pfopt_groups(self):
        assert pfopt.find_pfopt_groups(
            vbt.pfopt_group_dict({"a": (1, 2, 3), "d": (4, 5, 6)}),
            vbt.pfopt_group_dict({"b": (7, 8, 9), "_default": (10, 11, 12)}),
        ) == ["a", "d", "b"]
        assert pfopt.find_pfopt_groups(
            vbt.pfopt_group_dict({"a": (1, 2, 3), "d": (4, 5, 6)}),
            vbt.pfopt_group_dict({"b": (7, 8, 9), "_default": (10, 11, 12)}),
            sort_groups=True,
        ) == ["a", "b", "d"]
        assert pfopt.find_pfopt_groups(
            (
                vbt.pfopt_group_dict({"a": (1, 2, 3)}),
                vbt.pfopt_group_dict({"b": (4, 5, 6)}),
            ),
            {
                "x": vbt.pfopt_group_dict({"c": (7, 8, 9)}),
                "y": vbt.pfopt_group_dict({"_default": (10, 11, 12)}),
             },
        ) == ["a", "b", "c"]

    def test_select_pfopt_group_args(self):
        assert pfopt.select_pfopt_group_args(
            "a",
            (),
            {},
        ) == ((), {})
        assert pfopt.select_pfopt_group_args(
            "a",
            (1, 2, 3),
            {"k": (4, 5, 6)},
        ) == ((1, 2, 3), {"k": (4, 5, 6)})
        assert pfopt.select_pfopt_group_args(
            "a",
            vbt.pfopt_group_dict({"a": (1, 2, 3)}),
            vbt.pfopt_group_dict({"a": {"k": (4, 5, 6)}}),
        ) == ((1, 2, 3), {"k": (4, 5, 6)})
        assert pfopt.select_pfopt_group_args(
            "a",
            vbt.pfopt_group_dict({"b": (1, 2, 3)}),
            vbt.pfopt_group_dict({"b": {"k": (4, 5, 6)}}),
        ) == ((), {})
        assert pfopt.select_pfopt_group_args(
            "a",
            vbt.pfopt_group_dict({"_default": (1, 2, 3)}),
            vbt.pfopt_group_dict({"_default": {"k": (4, 5, 6)}}),
        ) == ((1, 2, 3), {"k": (4, 5, 6)})
        assert pfopt.select_pfopt_group_args(
            "a",
            (
                vbt.pfopt_group_dict({"a": 1}),
                vbt.pfopt_group_dict({"b": 2}),
                vbt.pfopt_group_dict({"_default": 3}),
            ),
            {
                "k1": vbt.pfopt_group_dict({"a": 4}),
                "k2": vbt.pfopt_group_dict({"b": 5}),
                "k3": vbt.pfopt_group_dict({"_default": 6}),
            }
        ) == ((1, 3), {"k1": 4, "k3": 6})
        assert pfopt.select_pfopt_group_args(
            "b",
            (
                vbt.pfopt_group_dict({"a": 1}),
                vbt.pfopt_group_dict({"b": 2}),
                vbt.pfopt_group_dict({"_default": 3}),
            ),
            {
                "k1": vbt.pfopt_group_dict({"a": 4}),
                "k2": vbt.pfopt_group_dict({"b": 5}),
                "k3": vbt.pfopt_group_dict({"_default": 6}),
            }
        ) == ((2, 3), {"k2": 5, "k3": 6})
        assert pfopt.select_pfopt_group_args(
            "c",
            (
                vbt.pfopt_group_dict({"a": 1}),
                vbt.pfopt_group_dict({"b": 2}),
                vbt.pfopt_group_dict({"_default": 3}),
            ),
            {
                "k1": vbt.pfopt_group_dict({"a": 4}),
                "k2": vbt.pfopt_group_dict({"b": 5}),
                "k3": vbt.pfopt_group_dict({"_default": 6}),
            }
        ) == ((3,), {"k3": 6})

    def test_from_optimize_func(self):
        def get_allocations(pf_opt):
            start_idx = pf_opt.alloc_records.values["start_idx"]
            end_idx = pf_opt.alloc_records.values["end_idx"]
            return np.array([
                prices.values[start_idx[i]:end_idx[i]].sum(axis=0)
                for i in range(len(pf_opt.alloc_records.values))
            ])

        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum().values,
            vbt.Rep("index_slice"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum().to_dict(),
            vbt.Rep("index_slice"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: {
                k: v
                for k, v in prices.iloc[index_slice].sum().to_dict().items()
                if k in prices.columns[:-1]
            },
            vbt.Rep("index_slice"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        allocations = get_allocations(pf_opt)
        allocations[:, -1] = np.nan
        np.testing.assert_array_equal(
            pf_opt._allocations,
            allocations,
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            every="2D"
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            every=vbt.RepEval("'2D'")
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=[(0, 2), (2, 4)],
            alloc_wait=0,
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 1, 1), (1, 0, 2, 4, 3, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=[(0, 2), (2, 4)],
            index_loc=[2, 4]
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.RepEval("[(0, 2), (2, 4)]"),
            index_loc=vbt.RepEval("[2, 4]")
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda i, index_ranges: prices.iloc[index_ranges[i, 0]:index_ranges[i, 1]].sum(),
            vbt.Rep("i"),
            every="2D",
            forward_args=["index_ranges"],
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda i, index_ranges: prices.iloc[index_ranges[i, 0]:index_ranges[i, 1]].sum(),
            vbt.Rep("i"),
            every="2D",
            forward_kwargs=["index_ranges"],
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            chunked=dict(arg_take_spec=dict(args=vbt.ArgsTaker(None))),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            jitted=dict(parallel=True),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            vbt.Rep("optimize_func"),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            template_context=dict(
                optimize_func=njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx]))
            ),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice, mult: prices.iloc[index_slice].sum() * mult,
            vbt.Rep("index_slice"),
            vbt.pfopt_group_dict({1: 1, 2: 2}),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0), (0, 1, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt) * np.array([[1], [2]]),
        )

        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice")
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.grouper.group_by,
            pd.Index(['group', 'group', 'group', 'group', 'group'], name="alloc_group"),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.columns,
            pd.MultiIndex.from_tuples([
                ('group', 'XOM'),
                ('group', 'RRC'),
                ('group', 'BBY'),
                ('group',  'MA'),
                ('group', 'PFE')],
                names=['alloc_group', None]
            ),
        )
        assert pf_opt.wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            pf_opt.alloc_records.wrapper.columns,
            pd.Index(["group"], name="alloc_group"),
        )
        assert pf_opt.alloc_records.wrapper.ndim == 1
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice, mult: prices.iloc[index_slice].sum() * mult,
            vbt.Rep("index_slice"),
            vbt.pfopt_group_dict({1: 1, 2: 2}),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.grouper.group_by,
            pd.Index([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], name="alloc_group"),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.columns,
            pd.MultiIndex.from_tuples([
                (1, 'XOM'),
                (1, 'RRC'),
                (1, 'BBY'),
                (1, 'MA'),
                (1, 'PFE'),
                (2, 'XOM'),
                (2, 'RRC'),
                (2, 'BBY'),
                (2, 'MA'),
                (2, 'PFE')],
                names=['alloc_group', None]
            ),
        )
        assert pf_opt.wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            pf_opt.alloc_records.wrapper.columns,
            pd.Index([1, 2], name="alloc_group"),
        )
        assert pf_opt.alloc_records.wrapper.ndim == 2

    def test_from_pypfopt(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.max_sharpe()
            np.testing.assert_array_equal(
                vbt.PortfolioOptimizer.from_pypfopt(prices=prices)._allocations[0],
                pd.Series(ef.clean_weights()).values,
            )

    def test_from_allocate_func(self):
        def get_allocations(pf_opt):
            idx = pf_opt.alloc_records.values["alloc_idx"]
            return np.array([
                prices.values[idx[i]]
                for i in range(len(pf_opt.alloc_records.values))
            ])

        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point].values,
            vbt.Rep("index_point"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point].to_dict(),
            vbt.Rep("index_point"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: {
                k: v
                for k, v in prices.iloc[index_point].to_dict().items()
                if k in prices.columns[:-1]
            },
            vbt.Rep("index_point"),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0)], dtype=alloc_point_dt),
        )
        allocations = get_allocations(pf_opt)
        allocations[:, -1] = np.nan
        np.testing.assert_array_equal(
            pf_opt._allocations,
            allocations,
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            every="2D"
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            every=vbt.RepEval("'2D'")
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            index_points=[2, 4]
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 2), (1, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            index_points=vbt.RepEval("[2, 4]")
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 2), (1, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda i, index_points: prices.iloc[index_points[i]],
            vbt.Rep("i"),
            every="2D",
            forward_args=["index_points"],
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda i, index_points: prices.iloc[index_points[i]],
            vbt.Rep("i"),
            every="2D",
            forward_kwargs=["index_points"],
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            chunked=dict(arg_take_spec=dict(args=vbt.ArgsTaker(None))),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            jitted=dict(parallel=True),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            vbt.Rep("allocate_func"),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            template_context=dict(
                allocate_func=njit(lambda i, idx, prices: prices[idx])
            ),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt),
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point, mult: prices.iloc[index_point] * mult,
            vbt.Rep("index_point"),
            vbt.pfopt_group_dict({1: 1, 2: 2}),
        )
        assert isinstance(pf_opt.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (0, 1, 0)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            get_allocations(pf_opt) * np.array([[1], [2]]),
        )

        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point")
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.grouper.group_by,
            pd.Index(['group', 'group', 'group', 'group', 'group'], name="alloc_group"),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.columns,
            pd.MultiIndex.from_tuples([
                ('group', 'XOM'),
                ('group', 'RRC'),
                ('group', 'BBY'),
                ('group',  'MA'),
                ('group', 'PFE')],
                names=['alloc_group', None]
            ),
        )
        assert pf_opt.wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            pf_opt.alloc_records.wrapper.columns,
            pd.Index(["group"], name="alloc_group"),
        )
        assert pf_opt.alloc_records.wrapper.ndim == 1
        pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point, mult: prices.iloc[index_point] * mult,
            vbt.Rep("index_point"),
            vbt.pfopt_group_dict({1: 1, 2: 2}),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.grouper.group_by,
            pd.Index([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], name="alloc_group"),
        )
        pd.testing.assert_index_equal(
            pf_opt.wrapper.columns,
            pd.MultiIndex.from_tuples([
                (1, 'XOM'),
                (1, 'RRC'),
                (1, 'BBY'),
                (1, 'MA'),
                (1, 'PFE'),
                (2, 'XOM'),
                (2, 'RRC'),
                (2, 'BBY'),
                (2, 'MA'),
                (2, 'PFE')],
                names=['alloc_group', None]
            ),
        )
        assert pf_opt.wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            pf_opt.alloc_records.wrapper.columns,
            pd.Index([1, 2], name="alloc_group"),
        )
        assert pf_opt.alloc_records.wrapper.ndim == 2

    def test_from_allocations(self):
        pf_opt = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.iloc[::2],
        )
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            prices.values[::2],
        )
        with pytest.raises(Exception):
            vbt.PortfolioOptimizer.from_allocations(
                prices.vbt.wrapper,
                prices.values[::2],
            )
        pf_opt = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.values[::2],
            every="2D",
        )
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            prices.values[::2],
        )
        pf_opt = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.iloc[::2].to_dict(orient="records"),
            every="2D",
        )
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            prices.values[::2],
        )

    def test_from_filled_allocations(self):
        filled_allocations = prices.copy()
        filled_allocations.iloc[1::2] = 0
        pf_opt = vbt.PortfolioOptimizer.from_filled_allocations(filled_allocations)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            prices.values[::2],
        )
        pf_opt = vbt.PortfolioOptimizer.from_filled_allocations(
            filled_allocations,
            notna_only=False,
            unique_only=False,
        )
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            filled_allocations,
        )
        with pytest.raises(Exception):
            vbt.PortfolioOptimizer.from_filled_allocations(
                prices.vbt.wrapper,
                filled_allocations.iloc[::2],
            )

    def test_from_uniform(self):
        pf_opt = vbt.PortfolioOptimizer.from_uniform(prices.vbt.wrapper, on=3)
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 3)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            np.array([[1/5, 1/5, 1/5, 1/5, 1/5]]),
        )

    def test_from_random(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=3,
        )
        assert_records_close(
            pf_opt.alloc_records.values,
            np.array([(0, 0, 3)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pf_opt._allocations,
            np.array([[
                0.13319702814025883,
                0.33810081711389406,
                0.26031768763785473,
                0.2128998389048247,
                0.05548462820316767,
            ]]),
        )

    def test_from_universal_algo(self):
        if universal_available:
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                "CRP",
                prices,
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                np.array([[1/5, 1/5, 1/5, 1/5, 1/5]]),
            )
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                "CRP",
                prices,
                notna_only=False,
                unique_only=False
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                np.array([
                    [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                    [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                    [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                    [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                    [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                ]),
            )
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP,
                prices,
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP(),
                prices,
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP().run(prices),
                wrapper=prices.vbt.wrapper,
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pf_opt = vbt.PortfolioOptimizer.from_universal_algo(
                "DynamicCRP",
                prices,
                n=52,
                min_history=8,
            )
            assert_records_close(
                pf_opt.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pf_opt._allocations,
                universal.algos.DynamicCRP(n=52, min_history=8).run(prices).weights.values[[0]],
            )

    def test_get_allocations(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=[1, 3],
        )
        target_arr = [[
            0.13319702814025883,
            0.33810081711389406,
            0.26031768763785473,
            0.2128998389048247,
            0.05548462820316767,
        ], [
            0.06528491964469331,
            0.02430844330237927,
            0.3625014516740258,
            0.2515713061862386,
            0.29633387919266296,
        ]]
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays([
                    pd.Index(["group", "group"], name="alloc_group"),
                    prices.index[[1, 3]],
                ]),
                columns=prices.columns,
            )
        )
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                target_arr,
                index=prices.index[[1, 3]],
                columns=prices.columns,
            )
        )
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=vbt.pfopt_group_dict({"g1": [1, 3], "g2": [2, 4]}),
        )
        target_arr = np.array([
            [0.13319702814025883, 0.33810081711389406, 0.26031768763785473, 0.2128998389048247, 0.05548462820316767],
            [0.06528491964469331, 0.02430844330237927, 0.3625014516740258, 0.2515713061862386, 0.29633387919266296],
            [0.00928441856775223, 0.4374675865753444, 0.37546445396096845, 0.09577331138241256, 0.0820102295135221],
            [0.10567348701744264, 0.17529742718559654, 0.302352663027335, 0.2488768479913802, 0.1677995747782457]
        ])
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays([
                    pd.Index(["g1", "g1", "g2", "g2"], name="alloc_group"),
                    prices.index[[1, 3, 2, 4]],
                ]),
                columns=prices.columns,
            )
        )
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays([
                    pd.Index(["g1", "g1", "g2", "g2"], name="alloc_group"),
                    prices.index[[1, 3, 2, 4]],
                ]),
                columns=prices.columns,
            )
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_loc=4,
        )
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                prices.sum().values[None],
                index=pd.MultiIndex.from_arrays([
                    pd.Index(["group"], name="alloc_group"),
                    prices.index[[4]],
                ]),
                columns=prices.columns,
            )
        )
        pd.testing.assert_frame_equal(
            pf_opt.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                prices.sum().values[None],
                index=prices.index[[4]],
                columns=prices.columns,
            )
        )

    def test_fill_allocations(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=[1, 3],
        )
        target_arr = [[
            0.13319702814025883,
            0.33810081711389406,
            0.26031768763785473,
            0.2128998389048247,
            0.05548462820316767,
        ], [
            0.06528491964469331,
            0.02430844330237927,
            0.3625014516740258,
            0.2515713061862386,
            0.29633387919266296,
        ]]
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays([
                pd.Index(["group", "group", "group", "group", "group"], name="alloc_group"),
                prices.columns,
            ]),
        )
        target_df.iloc[[1, 3]] = target_arr
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=False),
            target_df,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=prices.columns,
        )
        target_df.iloc[[1, 3]] = target_arr
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=True),
            target_df,
        )
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=vbt.pfopt_group_dict({"g1": [1, 3], "g2": [2, 4]}),
        )
        target_arr = np.array([
            [0.13319702814025883, 0.33810081711389406, 0.26031768763785473, 0.2128998389048247, 0.05548462820316767],
            [0.06528491964469331, 0.02430844330237927, 0.3625014516740258, 0.2515713061862386, 0.29633387919266296],
            [0.00928441856775223, 0.4374675865753444, 0.37546445396096845, 0.09577331138241256, 0.0820102295135221],
            [0.10567348701744264, 0.17529742718559654, 0.302352663027335, 0.2488768479913802, 0.1677995747782457]
        ])
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays([
                pd.Index(["g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g2"], name="alloc_group"),
                prices.columns.append(prices.columns),
            ]),
        )
        target_df.iloc[[1, 3], 0:5] = target_arr[0:2]
        target_df.iloc[[2, 4], 5:10] = target_arr[2:4]
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=False),
            target_df
        )
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=True),
            target_df
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_loc=4,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays([
                pd.Index(["group", "group", "group", "group", "group"], name="alloc_group"),
                prices.columns,
            ]),
        )
        target_df.iloc[[4]] = prices.sum().values
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=False),
            target_df,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=prices.columns,
        )
        target_df.iloc[[4]] = prices.sum().values
        pd.testing.assert_frame_equal(
            pf_opt.fill_allocations(squeeze_groups=True),
            target_df,
        )

    def test_points_stats(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=vbt.pfopt_group_dict({"g1": [1, 3], "g2": [2, 4]}),
        )
        stats_index = pd.Index([
            'Start',
            'End',
            'Period',
            'Total Records',
            'Mean Allocation: XOM',
            'Mean Allocation: RRC',
            'Mean Allocation: BBY',
            'Mean Allocation: MA',
            'Mean Allocation: PFE',
        ], dtype='object')
        pd.testing.assert_series_equal(
            pf_opt.stats(),
            pd.Series(
                [
                    pd.Timestamp('2020-01-01 00:00:00', freq='D'),
                    pd.Timestamp('2020-01-05 00:00:00', freq='D'),
                    pd.Timedelta('5 days 00:00:00'),
                    2.0,
                    0.07835996334253675,
                    0.24379356854430356,
                    0.32515906407504597,
                    0.202280326116214,
                    0.1504070779218996,
                ],
                index=stats_index,
                name="agg_stats"
            )
        )
        pd.testing.assert_series_equal(
            pf_opt.stats(column="g1"),
            pd.Series(
                [
                    pd.Timestamp('2020-01-01 00:00:00', freq='D'),
                    pd.Timestamp('2020-01-05 00:00:00', freq='D'),
                    pd.Timedelta('5 days 00:00:00'),
                    2,
                    0.09924097389247608,
                    0.18120463020813665,
                    0.31140956965594024,
                    0.23223557254553162,
                    0.17590925369791532,
                ],
                index=stats_index,
                name="g1"
            )
        )
        pd.testing.assert_series_equal(pf_opt["g1"].stats(), pf_opt.stats(column="g1"))
        stats_df = pf_opt.stats(agg_func=None)
        assert stats_df.shape == (2, 9)
        pd.testing.assert_index_equal(stats_df.index, pf_opt.wrapper.get_columns())
        pd.testing.assert_index_equal(stats_df.columns, stats_index)

    def test_ranges_stats(self):
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.pfopt_group_dict({"g1": [0, 2], "g2": [1, 3]}),
        )
        stats_index = pd.Index([
            'Start',
            'End',
            'Period',
            'Total Records',
            'Coverage',
            'Overlap Coverage',
            'Mean Allocation: XOM',
            'Mean Allocation: RRC',
            'Mean Allocation: BBY',
            'Mean Allocation: MA',
            'Mean Allocation: PFE',
        ], dtype='object')
        pd.testing.assert_series_equal(
            pf_opt.stats(),
            pd.Series(
                [
                    pd.Timestamp('2020-01-01 00:00:00', freq='D'),
                    pd.Timestamp('2020-01-05 00:00:00', freq='D'),
                    pd.Timedelta('5 days 00:00:00'),
                    1.0,
                    0.4,
                    0.0,
                    108.68882550000001,
                    103.4886705,
                    66.15678550000001,
                    44.069272,
                    27.5600615,
                ],
                index=stats_index,
                name="agg_stats"
            )
        )
        pd.testing.assert_series_equal(
            pf_opt.stats(column="g1"),
            pd.Series(
                [
                    pd.Timestamp('2020-01-01 00:00:00', freq='D'),
                    pd.Timestamp('2020-01-05 00:00:00', freq='D'),
                    pd.Timedelta('5 days 00:00:00'),
                    1,
                    0.4,
                    0.0,
                    108.348701,
                    103.293606,
                    65.873542,
                    44.059574999999995,
                    27.681569,
                ],
                index=stats_index,
                name="g1"
            )
        )
        pd.testing.assert_series_equal(pf_opt["g1"].stats(), pf_opt.stats(column="g1"))
        stats_df = pf_opt.stats(agg_func=None)
        assert stats_df.shape == (2, 11)
        pd.testing.assert_index_equal(stats_df.index, pf_opt.wrapper.get_columns())
        pd.testing.assert_index_equal(stats_df.columns, stats_index)

    def test_plots(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=vbt.pfopt_group_dict({"g1": [1, 3], "g2": [2, 4]}),
        )
        pf_opt["g1"].plot()
        pf_opt[["g1"]].plot()
        pf_opt.plot(column="g1")

        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.pfopt_group_dict({"g1": [0, 2], "g2": [1, 3]}),
        )
        pf_opt["g1"].plot()
        pf_opt[["g1"]].plot()
        pf_opt.plot(column="g1")


# ############# Portfolio ############# #

class TestPortfolio:
    def test_from_optimizer(self):
        pf_opt = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=vbt.pfopt_group_dict({"g1": [1, 3], "g2": [2, 4]}),
        )
        pf = vbt.Portfolio.from_optimizer(pf_opt, prices)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[1][:5],
            pf_opt.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][:5],
            pf_opt.allocations.values[1],
        )
        np.testing.assert_allclose(
            allocations.values[2][5:10],
            pf_opt.allocations.values[2],
        )
        np.testing.assert_allclose(
            allocations.values[4][5:10],
            pf_opt.allocations.values[3],
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum() / prices.iloc[index_slice].sum().sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.pfopt_group_dict({"g1": [0, 2], "g2": [1, 3]}),
        )
        pf = vbt.Portfolio.from_optimizer(pf_opt, prices)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pf_opt.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pf_opt.allocations.values[1],
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.pfopt_group_dict({"g1": [0, 2], "g2": [1, 3]}),
        )
        pf = vbt.Portfolio.from_optimizer(pf_opt, prices, init_cash="auto")
        allocations = pf.asset_flow
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pf_opt.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pf_opt.allocations.values[1],
        )
        pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: -prices.iloc[index_slice].sum() / prices.iloc[index_slice].sum().sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.pfopt_group_dict({"g1": [0, 2], "g2": [1, 3]}),
        )
        pf = vbt.Portfolio.from_optimizer(pf_opt, prices)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pf_opt.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pf_opt.allocations.values[1],
        )
