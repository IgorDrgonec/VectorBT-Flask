import os

import numpy as np
import pandas as pd

import vectorbtpro as vbt
from vectorbtpro.portfolio import pfopt

pypfopt_available = True
try:
    import pypfopt
except:
    pypfopt_available = False

# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# pfopt ############# #

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


class TestPfOpt:
    def test_resolve_expected_returns(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns("mean_historical_return", prices=prices),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns("mean_historical_return", prices=prices, returns_data=False),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns("mean_historical_return", prices=returns, returns_data=True),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns("mean_historical_return", returns=returns),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns("mean_historical_return", prices=prices, returns=returns),
                pypfopt.expected_returns.mean_historical_return(prices, returns_data=False),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns(
                    "mean_historical_return",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                ),
                pypfopt.expected_returns.mean_historical_return(prices, frequency=365, compounding=False),
            )
            pd.testing.assert_series_equal(
                pfopt.resolve_expected_returns(
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
                pfopt.resolve_expected_returns(
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
                pfopt.resolve_expected_returns(
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
                pfopt.resolve_expected_returns(
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
                pfopt.resolve_expected_returns(
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

    def test_resolve_cov_matrix(self):
        if pypfopt_available:
            np.testing.assert_array_equal(
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
                    "ledoit_wolf",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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
                pfopt.resolve_cov_matrix(
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

    def test_resolve_optimizer(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            weights1 = pfopt.resolve_optimizer("efficient_frontier", prices=prices).min_volatility()
            weights2 = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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

            weights1 = pfopt.resolve_optimizer(
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
            weights1 = pfopt.resolve_optimizer(
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

    def test_get_allocation(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.max_sharpe()
            assert pfopt.get_allocation(prices=prices) == ef.clean_weights()
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.ema_historical_return(prices=prices),
                pypfopt.risk_models.sample_cov(prices=prices),
            )
            ef.max_sharpe()
            assert (
                pfopt.get_allocation(
                    prices=prices,
                    expected_returns="ema_historical_return",
                    cov_matrix="sample_cov",
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
                pfopt.get_allocation(
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
                pfopt.get_allocation(
                    prices=prices,
                    objectives="L2_reg",
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.get_allocation(
                    prices=prices,
                    objectives=["L2_reg"],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.get_allocation(
                    prices=prices,
                    objectives=objective_functions.L2_reg,
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.get_allocation(
                    prices=prices,
                    objectives=[objective_functions.L2_reg],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

            da = DiscreteAllocation(ef.clean_weights(), get_latest_prices(prices), total_portfolio_value=20000)
            assert (
                pfopt.get_allocation(
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
                pfopt.get_allocation(
                    prices=prices,
                    constraints=lambda w: w[0] >= 0.2,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.get_allocation(
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
                pfopt.get_allocation(
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
                pfopt.get_allocation(
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
                pfopt.get_allocation(
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
                pfopt.get_allocation(
                    prices=prices,
                    target=deviation_risk_parity,
                    target_is_convex=False,
                    weight_bounds=(0.01, 0.12),
                )
                == ef.clean_weights()
            )
