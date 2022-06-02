# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with log records.

Order records capture information on simulation logs. Logs are populated when
simulating a portfolio and can be accessed as `vectorbtpro.portfolio.base.Portfolio.logs`.

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbtpro as vbt

>>> np.random.seed(42)
>>> price = pd.DataFrame({
...     'a': np.random.uniform(1, 2, size=100),
...     'b': np.random.uniform(1, 2, size=100)
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> size = pd.DataFrame({
...     'a': np.random.uniform(-100, 100, size=100),
...     'b': np.random.uniform(-100, 100, size=100),
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d', log=True)
>>> logs = pf.logs

>>> logs.filled.count()
a    88
b    99
Name: count, dtype: int64

>>> logs.ignored.count()
a    0
b    0
Name: count, dtype: int64

>>> logs.rejected.count()
a    12
b     1
Name: count, dtype: int64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Logs.metrics`.

```pycon
>>> logs['a'].stats()
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     100
Status Counts: None                                 0
Status Counts: Filled                              88
Status Counts: Ignored                              0
Status Counts: Rejected                            12
Status Info Counts: None                           88
Status Info Counts: NoCashLong                     12
Name: a, dtype: object
```

`Logs.stats` also supports (re-)grouping:

```pycon
>>> logs.stats(group_by=True)
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     200
Status Counts: None                                 0
Status Counts: Filled                             187
Status Counts: Ignored                              0
Status Counts: Rejected                            13
Status Info Counts: None                          187
Status Info Counts: NoCashLong                     13
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Logs.subplots`.

This class does not have any subplots.
"""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_dict
from vectorbtpro.generic.price_records import PriceRecords
from vectorbtpro.portfolio.enums import log_dt, SizeType, Direction, OrderSide, OrderStatus, OrderStatusInfo
from vectorbtpro.records.decorators import attach_fields, override_field_config
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig

__pdoc__ = {}

logs_field_config = ReadonlyConfig(
    dict(
        dtype=log_dt,
        settings=dict(
            id=dict(title=("Meta", "Log Id")),
            col=dict(title=("Meta", "Column")),
            idx=dict(title=("Meta", "Timestamp")),
            group=dict(title=("Meta", "Group")),
            price_area_open=dict(title=("Price Area", "Open")),
            price_area_high=dict(title=("Price Area", "High")),
            price_area_low=dict(title=("Price Area", "Low")),
            price_area_close=dict(title=("Price Area", "Close")),
            exec_state_cash=dict(title=("Exec State", "Cash")),
            exec_state_position=dict(title=("Exec State", "Position")),
            exec_state_debt=dict(title=("Exec State", "Debt")),
            exec_state_free_cash=dict(title=("Exec State", "Free Cash")),
            exec_state_val_price=dict(title=("Exec State", "Valuation Price")),
            exec_state_value=dict(title=("Exec State", "Value")),
            req_size=dict(title=("Request", "Size")),
            req_price=dict(title=("Request", "Price")),
            req_size_type=dict(title=("Request", "Size Type"), mapping=SizeType),
            req_direction=dict(title=("Request", "Direction"), mapping=Direction),
            req_fees=dict(title=("Request", "Fees")),
            req_fixed_fees=dict(title=("Request", "Fixed Fees")),
            req_slippage=dict(title=("Request", "Slippage")),
            req_min_size=dict(title=("Request", "Min Size")),
            req_max_size=dict(title=("Request", "Max Size")),
            req_size_granularity=dict(title=("Request", "Size Granularity")),
            req_reject_prob=dict(title=("Request", "Rejection Prob")),
            req_price_area_vio_mode=dict(title=("Request", "Price Area Violation Mode")),
            req_lock_cash=dict(title=("Request", "Lock Cash")),
            req_allow_partial=dict(title=("Request", "Allow Partial")),
            req_raise_reject=dict(title=("Request", "Raise Rejection")),
            req_log=dict(title=("Request", "Log")),
            new_exec_state_cash=dict(title=("New Exec State", "Cash")),
            new_exec_state_position=dict(title=("New Exec State", "Position")),
            new_exec_state_debt=dict(title=("New Exec State", "Debt")),
            new_exec_state_free_cash=dict(title=("New Exec State", "Free Cash")),
            new_exec_state_val_price=dict(title=("New Exec State", "Valuation Price")),
            new_exec_state_value=dict(title=("New Exec State", "Value")),
            res_size=dict(title=("Result", "Size")),
            res_price=dict(title=("Result", "Price")),
            res_fees=dict(title=("Result", "Fees")),
            res_side=dict(title=("Result", "Side"), mapping=OrderSide),
            res_status=dict(title=("Result", "Status"), mapping=OrderStatus),
            res_status_info=dict(title=("Result", "Status Info"), mapping=OrderStatusInfo),
            order_id=dict(title=("Result", "Order Id"), mapping="ids"),
        ),
    )
)
"""_"""

__pdoc__[
    "logs_field_config"
] = f"""Field config for `Logs`.

```python
{logs_field_config.prettify()}
```
"""

logs_attach_field_config = ReadonlyConfig(
    dict(
        res_side=dict(attach_filters=True),
        res_status=dict(attach_filters=True),
        res_status_info=dict(attach_filters=True),
    )
)
"""_"""

__pdoc__[
    "logs_attach_field_config"
] = f"""Config of fields to be attached to `Logs`.

```python
{logs_attach_field_config.prettify()}
```
"""

LogsT = tp.TypeVar("LogsT", bound="Logs")


@attach_fields(logs_attach_field_config)
@override_field_config(logs_field_config)
class Logs(PriceRecords):
    """Extends `vectorbtpro.generic.price_records.PriceRecords` for working with log records."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Logs.stats`.

        Merges `vectorbtpro.generic.price_records.PriceRecords.stats_defaults` and
        `stats` from `vectorbtpro._settings.logs`."""
        from vectorbtpro._settings import settings

        logs_stats_cfg = settings["logs"]["stats"]

        return merge_dicts(PriceRecords.stats_defaults.__get__(self), logs_stats_cfg)

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
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            res_status_counts=dict(
                title="Status Counts",
                calc_func="res_status.value_counts",
                incl_all_keys=True,
                post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
                tags=["logs", "res_status", "value_counts"],
            ),
            res_status_info_counts=dict(
                title="Status Info Counts",
                calc_func="res_status_info.value_counts",
                post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
                tags=["logs", "res_status_info", "value_counts"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Logs.plots`.

        Merges `vectorbtpro.generic.price_records.PriceRecords.plots_defaults` and
        `plots` from `vectorbtpro._settings.logs`."""
        from vectorbtpro._settings import settings

        logs_plots_cfg = settings["logs"]["plots"]

        return merge_dicts(PriceRecords.plots_defaults.__get__(self), logs_plots_cfg)

    @property
    def subplots(self) -> Config:
        return self._subplots


Logs.override_field_config_doc(__pdoc__)
Logs.override_metrics_doc(__pdoc__)
Logs.override_subplots_doc(__pdoc__)
