# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with trade records.

Trade records capture information on trades.

In vectorbt, a trade is a sequence of orders that starts with an opening order and optionally ends
with a closing order. Every pair of opposite orders can be represented by a trade. Each trade has a PnL
info attached to quickly assess its performance. An interesting effect of this representation
is the ability to aggregate trades: if two or more trades are happening one after another in time,
they can be aggregated into a bigger trade. This way, for example, single-order trades can be aggregated
into positions; but also multiple positions can be aggregated into a single blob that reflects the performance
of the entire symbol.

!!! warning
    All classes return both closed AND open trades/positions, which may skew your performance results.
    To only consider closed trades/positions, you should explicitly query the `status_closed` attribute.

## Trade types

There are three main types of trades.

### Entry trades

An entry trade is created from each order that opens or adds to a position.

For example, if we have a single large buy order and 100 smaller sell orders, we will see
a single trade with the entry information copied from the buy order and the exit information being
a size-weighted average over the exit information of all sell orders. On the other hand,
if we have 100 smaller buy orders and a single sell order, we will see 100 trades,
each with the entry information copied from the buy order and the exit information being
a size-based fraction of the exit information of the sell order.

Use `vectorbtpro.portfolio.trades.EntryTrades.from_orders` to build entry trades from orders.
Also available as `vectorbtpro.portfolio.base.Portfolio.entry_trades`.

### Exit trades

An exit trade is created from each order that closes or removes from a position.

Use `vectorbtpro.portfolio.trades.ExitTrades.from_orders` to build exit trades from orders.
Also available as `vectorbtpro.portfolio.base.Portfolio.exit_trades`.

### Positions

A position is created from a sequence of entry or exit trades.

Use `vectorbtpro.portfolio.trades.Positions.from_trades` to build positions from entry or exit trades.
Also available as `vectorbtpro.portfolio.base.Portfolio.positions`.

## Example

* Increasing position:

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbtpro as vbt

>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 1., 1., 1., -4.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         1.0
2         2       0   1.0                2              3.0         1.0
3         3       0   1.0                3              4.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees   PnL  Return Direction  Status  \\
0               4             5.0       0.25  2.75  2.7500      Long  Closed
1               4             5.0       0.25  1.75  0.8750      Long  Closed
2               4             5.0       0.25  0.75  0.2500      Long  Closed
3               4             5.0       0.25 -0.25 -0.0625      Long  Closed

   Parent Id
0          0
1          0
2          0
3          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              2.5         4.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             5.0        1.0  5.0     0.5      Long  Closed

   Parent Id
0          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              2.5         4.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             5.0        1.0  5.0     0.5      Long  Closed

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Decreasing position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([4., -1., -1., -1., -1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             3.5        4.0  5.0    1.25      Long  Closed

   Parent Id
0          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0        0.25
1         1       0   1.0                0              1.0        0.25
2         2       0   1.0                0              1.0        0.25
3         3       0   1.0                0              1.0        0.25

   Exit Timestamp  Avg Exit Price  Exit Fees   PnL  Return Direction  Status  \\
0               1             2.0        1.0 -0.25   -0.25      Long  Closed
1               2             3.0        1.0  0.75    0.75      Long  Closed
2               3             4.0        1.0  1.75    1.75      Long  Closed
3               4             5.0        1.0  2.75    2.75      Long  Closed

   Parent Id
0          0
1          0
2          0
3          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             3.5        4.0  5.0    1.25      Long  Closed

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Multiple reversing positions:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., -2., 2., -2., 1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Open position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 0., 0., 0., 0.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Get trade count, trade PnL, and winning trade PnL:

```pycon
>>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
>>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
>>> trades = vbt.Portfolio.from_orders(price, size).trades

>>> trades.count()
6

>>> trades.pnl.sum()
-3.0

>>> trades.winning.count()
2

>>> trades.winning.pnl.sum()
1.5
```

Get count and PnL of trades with duration of more than 2 days:

```pycon
>>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
>>> trades_filtered = trades.apply_mask(mask)
>>> trades_filtered.count()
2

>>> trades_filtered.pnl.sum()
-3.0
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Trades.metrics`.

```pycon
>>> price = vbt.RandomData.fetch(
...     ['a', 'b'],
...     start=datetime(2020, 1, 1),
...     end=datetime(2020, 3, 1),
...     seed=vbt.symbol_dict(a=42, b=43)
... ).get()
```

[=100% "100%"]{: .candystripe}

```pycon
>>> size = pd.DataFrame({
...     'a': np.random.randint(-1, 2, size=len(price.index)),
...     'b': np.random.randint(-1, 2, size=len(price.index)),
... }, index=price.index, columns=price.columns)
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d')

>>> pf.trades['a'].stats()
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                60 days 00:00:00
Overlap Coverage                        49 days 00:00:00
Total Records                                       19.0
Total Long Trades                                    2.0
Total Short Trades                                  17.0
Total Closed Trades                                 18.0
Total Open Trades                                    1.0
Open Trade PnL                                    16.063
Win Rate [%]                                   61.111111
Max Win Streak                                      11.0
Max Loss Streak                                      7.0
Best Trade [%]                                  3.526377
Worst Trade [%]                                -6.543679
Avg Winning Trade [%]                           2.225861
Avg Losing Trade [%]                           -3.601313
Avg Winning Trade Duration    32 days 19:38:10.909090909
Avg Losing Trade Duration                5 days 00:00:00
Profit Factor                                   1.022425
Expectancy                                      0.028157
SQN                                             0.039174
Name: agg_stats, dtype: object
```

Positions share almost identical metrics with trades:

```pycon
>>> pf.positions['a'].stats()
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                        0 days 00:00:00
Total Records                                       5.0
Total Long Trades                                   2.0
Total Short Trades                                  3.0
Total Closed Trades                                 4.0
Total Open Trades                                   1.0
Open Trade PnL                                38.356823
Win Rate [%]                                        0.0
Max Win Streak                                      0.0
Max Loss Streak                                     4.0
Best Trade [%]                                -1.529613
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                               NaN
Avg Losing Trade [%]                          -3.786739
Avg Winning Trade Duration                          NaT
Avg Losing Trade Duration               4 days 00:00:00
Profit Factor                                       0.0
Expectancy                                    -5.446748
SQN                                           -1.794214
Name: agg_stats, dtype: object
```

To also include open trades/positions when calculating metrics such as win rate, pass `incl_open=True`:

```pycon
>>> pf.trades['a'].stats(settings=dict(incl_open=True))
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                       49 days 00:00:00
Total Records                                      19.0
Total Long Trades                                   2.0
Total Short Trades                                 17.0
Total Closed Trades                                18.0
Total Open Trades                                   1.0
Open Trade PnL                                   16.063
Win Rate [%]                                  61.111111
Max Win Streak                                     12.0
Max Loss Streak                                     7.0
Best Trade [%]                                 3.526377
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                          2.238896
Avg Losing Trade [%]                          -3.601313
Avg Winning Trade Duration             33 days 18:00:00
Avg Losing Trade Duration               5 days 00:00:00
Profit Factor                                  1.733143
Expectancy                                     0.872096
SQN                                            0.804714
Name: agg_stats, dtype: object
```

`Trades.stats` also supports (re-)grouping:

```pycon
>>> pf.trades.stats(group_by=True)
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                61 days 00:00:00
Overlap Coverage                        61 days 00:00:00
Total Records                                         37
Total Long Trades                                      5
Total Short Trades                                    32
Total Closed Trades                                   35
Total Open Trades                                      2
Open Trade PnL                                  1.336259
Win Rate [%]                                   37.142857
Max Win Streak                                        11
Max Loss Streak                                       10
Best Trade [%]                                  3.526377
Worst Trade [%]                                -8.710238
Avg Winning Trade [%]                           1.907799
Avg Losing Trade [%]                           -3.259135
Avg Winning Trade Duration    28 days 14:46:09.230769231
Avg Losing Trade Duration               14 days 00:00:00
Profit Factor                                   0.340493
Expectancy                                     -1.292596
SQN                                            -2.509223
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Trades.subplots`.

`Trades` class has two subplots based on `Trades.plot` and `Trades.plot_pnl`:

```pycon
>>> pf.trades['a'].plots(settings=dict(plot_zones=False)).show()
```

![](/assets/images/api/trades_plots.svg)
"""

from functools import partialmethod

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.generic.enums import range_dt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import TradeDirection, TradeStatus, trade_dt
from vectorbtpro.portfolio.orders import Orders
from vectorbtpro.records.decorators import attach_fields, override_field_config, attach_shortcut_properties
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.array_ import min_rel_rescale, max_rel_rescale
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.template import RepEval

__pdoc__ = {}

# ############# Trades ############# #

trades_field_config = ReadonlyConfig(
    dict(
        dtype=trade_dt,
        settings={
            "id": dict(title="Trade Id"),
            "idx": dict(name="exit_idx"),  # remap field of Records
            "start_idx": dict(name="entry_idx"),  # remap field of Ranges
            "end_idx": dict(name="exit_idx"),  # remap field of Ranges
            "size": dict(title="Size"),
            "entry_idx": dict(title="Entry Index", mapping="index"),
            "entry_price": dict(title="Avg Entry Price"),
            "entry_fees": dict(title="Entry Fees"),
            "exit_idx": dict(title="Exit Index", mapping="index"),
            "exit_price": dict(title="Avg Exit Price"),
            "exit_fees": dict(title="Exit Fees"),
            "pnl": dict(title="PnL"),
            "return": dict(title="Return", hovertemplate="$title: %{customdata[$index]:,%}"),
            "direction": dict(title="Direction", mapping=TradeDirection),
            "status": dict(title="Status", mapping=TradeStatus),
            "parent_id": dict(title="Position Id", mapping="ids"),
        },
    )
)
"""_"""

__pdoc__[
    "trades_field_config"
] = f"""Field config for `Trades`.

```python
{trades_field_config.prettify()}
```
"""

trades_attach_field_config = ReadonlyConfig(
    {
        "return": dict(attach="returns"),
        "direction": dict(attach_filters=True),
        "status": dict(attach_filters=True, on_conflict="ignore"),
    }
)
"""_"""

__pdoc__[
    "trades_attach_field_config"
] = f"""Config of fields to be attached to `Trades`.

```python
{trades_attach_field_config.prettify()}
```
"""

trades_shortcut_config = ReadonlyConfig(
    dict(
        ranges=dict(),
        winning=dict(),
        losing=dict(),
        winning_streak=dict(obj_type="mapped_array"),
        losing_streak=dict(obj_type="mapped_array"),
        win_rate=dict(obj_type="red_array"),
        profit_factor=dict(obj_type="red_array"),
        expectancy=dict(obj_type="red_array"),
        sqn=dict(obj_type="red_array"),
        best_price=dict(obj_type="mapped_array"),
        worst_price=dict(obj_type="mapped_array"),
        mfe=dict(obj_type="mapped_array"),
        mae=dict(obj_type="mapped_array"),
        mfe_returns=dict(
            obj_type="mapped_array",
            method_name="get_mfe",
            method_kwargs=dict(as_returns=True),
        ),
        mae_returns=dict(
            obj_type="mapped_array",
            method_name="get_mae",
            method_kwargs=dict(as_returns=True),
        ),
    )
)
"""_"""

__pdoc__[
    "trades_shortcut_config"
] = f"""Config of shortcut properties to be attached to `Trades`.

```python
{trades_shortcut_config.prettify()}
```
"""

TradesT = tp.TypeVar("TradesT", bound="Trades")


@attach_shortcut_properties(trades_shortcut_config)
@attach_fields(trades_attach_field_config)
@override_field_config(trades_field_config)
class Trades(Ranges):
    """Extends `vectorbtpro.generic.ranges.Ranges` for working with trade-like records, such as
    entry trades, exit trades, and positions."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    def get_ranges(self, **kwargs) -> Ranges:
        """Get records of type `vectorbtpro.generic.ranges.Ranges`."""
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("entry_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("exit_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    # ############# Stats ############# #

    def get_winning(self: TradesT, **kwargs) -> TradesT:
        """Get winning trades."""
        filter_mask = self.values["pnl"] > 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_losing(self: TradesT, **kwargs) -> TradesT:
        """Get losing trades."""
        filter_mask = self.values["pnl"] < 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_winning_streak(self, **kwargs) -> MappedArray:
        """Get winning streak at each trade in the current column.

        See `vectorbtpro.portfolio.nb.records.trade_winning_streak_nb`."""
        return self.apply(nb.trade_winning_streak_nb, dtype=np.int_, **kwargs)

    def get_losing_streak(self, **kwargs) -> MappedArray:
        """Get losing streak at each trade in the current column.

        See `vectorbtpro.portfolio.nb.records.trade_losing_streak_nb`."""
        return self.apply(nb.trade_losing_streak_nb, dtype=np.int_, **kwargs)

    def get_win_rate(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get rate of winning trades."""
        wrap_kwargs = merge_dicts(dict(name_or_index="win_rate"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.win_rate_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_profit_factor(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get profit factor."""
        wrap_kwargs = merge_dicts(dict(name_or_index="profit_factor"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.profit_factor_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_expectancy(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get average profitability."""
        wrap_kwargs = merge_dicts(dict(name_or_index="expectancy"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.expectancy_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_sqn(
        self,
        ddof: int = 1,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get System Quality Number (SQN)."""
        wrap_kwargs = merge_dicts(dict(name_or_index="sqn"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.sqn_reduce_nb,
            ddof,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_best_price(self, entry_price_open: bool = False, exit_price_close: bool = False, **kwargs) -> MappedArray:
        """Get best price.

        See `vectorbtpro.portfolio.nb.records.best_worst_price_nb`."""
        return self.apply(
            nb.best_worst_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            True,
            **kwargs,
        )

    def get_worst_price(self, entry_price_open: bool = False, exit_price_close: bool = False, **kwargs) -> MappedArray:
        """Get worst price.

        See `vectorbtpro.portfolio.nb.records.best_worst_price_nb`."""
        return self.apply(
            nb.best_worst_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            False,
            **kwargs,
        )

    def get_mfe(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        as_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get MFE.

        See `vectorbtpro.portfolio.nb.records.mfe_nb`."""
        best_price = self.resolve_shortcut_attr(
            "best_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mfe_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        drawdown = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            best_price.values,
            as_returns=as_returns,
        )
        return self.map_array(drawdown, **kwargs)

    def get_mae(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        as_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get MAE.

        See `vectorbtpro.portfolio.nb.records.mae_nb`."""
        worst_price = self.resolve_shortcut_attr(
            "worst_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mae_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        drawdown = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            worst_price.values,
            as_returns=as_returns,
        )
        return self.map_array(drawdown, **kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.stats`.

        Merges `vectorbtpro.generic.ranges.Ranges.stats_defaults` and
        `stats` from `vectorbtpro._settings.trades`."""
        from vectorbtpro._settings import settings

        trades_stats_cfg = settings["trades"]["stats"]

        return merge_dicts(Ranges.stats_defaults.__get__(self), trades_stats_cfg)

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
            first_trade_start=dict(
                title="First Trade Start",
                calc_func="entry_idx.nth",
                n=0,
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            last_trade_end=dict(
                title="Last Trade End",
                calc_func="exit_idx.nth",
                n=-1,
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            total_long_trades=dict(
                title="Total Long Trades", calc_func="direction_long.count", tags=["trades", "long"]
            ),
            total_short_trades=dict(
                title="Total Short Trades", calc_func="direction_short.count", tags=["trades", "short"]
            ),
            total_closed_trades=dict(
                title="Total Closed Trades", calc_func="status_closed.count", tags=["trades", "closed"]
            ),
            total_open_trades=dict(title="Total Open Trades", calc_func="status_open.count", tags=["trades", "open"]),
            open_trade_pnl=dict(title="Open Trade PnL", calc_func="status_open.pnl.sum", tags=["trades", "open"]),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="status_closed.get_win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            winning_streak=dict(
                title="Max Win Streak",
                calc_func=RepEval("'winning_streak.max' if incl_open else 'status_closed.winning_streak.max'"),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            losing_streak=dict(
                title="Max Loss Streak",
                calc_func=RepEval("'losing_streak.max' if incl_open else 'status_closed.losing_streak.max'"),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func=RepEval("'returns.max' if incl_open else 'status_closed.returns.max'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func=RepEval("'returns.min' if incl_open else 'status_closed.returns.min'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func=RepEval("'winning.returns.mean' if incl_open else 'status_closed.winning.returns.mean'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func=RepEval("'losing.returns.mean' if incl_open else 'status_closed.losing.returns.mean'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func=RepEval("'winning.avg_duration' if incl_open else 'status_closed.winning.get_avg_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func=RepEval("'losing.avg_duration' if incl_open else 'status_closed.losing.get_avg_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func=RepEval("'profit_factor' if incl_open else 'status_closed.get_profit_factor'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func=RepEval("'expectancy' if incl_open else 'status_closed.get_expectancy'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            sqn=dict(
                title="SQN",
                calc_func=RepEval("'sqn' if incl_open else 'status_closed.get_sqn'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_pnl(
        self,
        column: tp.Optional[tp.Label] = None,
        pct_scale: bool = True,
        marker_size_range: tp.Tuple[float, float] = (7, 14),
        opacity_range: tp.Tuple[float, float] = (0.75, 0.9),
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot trade PnL or returns.

        Args:
            column (str): Name of the column to plot.
            pct_scale (bool): Whether to set y-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            marker_size_range (tuple): Range of marker size.
            opacity_range (tuple): Range of marker opacity.
            closed_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed" markers.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_pnl()
            ```

            ![](/assets/images/api/trades_plot_pnl.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        marker_size_range = tuple(marker_size_range)
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
            def_layout_kwargs[yaxis]["title"] = "Return"
        else:
            def_layout_kwargs[yaxis]["title"] = "PnL"
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=exit_idx[mask],
                            y=returns[mask] if pct_scale else pnl[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=marker_size[mask],
                                opacity=opacity[mask],
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(neutral_mask, "Closed", plotting_cfg["contrast_color_schema"]["gray"], closed_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(open_mask, "Open", plotting_cfg["contrast_color_schema"]["orange"], open_trace_kwargs)

        # Plot zeroline
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    plot_returns = partialmethod(plot_pnl, pct_scale=True)
    """`Trades.plot_pnl` for `Trades.returns`."""

    def plot_against_pnl(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Label] = None,
        pct_scale: bool = True,
        field_pct_scale: bool = False,
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        vline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a field against PnL or returns.

        Args:
            field (str, MappedArray, or array_like): Field to be plotted.
            field_label (str): Label of the field to be displayed on hover.
            column (str): Name of the column to plot.
            pct_scale (bool): Whether to set x-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            field_pct_scale (bool): Whether to make y-axis a percentage scale.
            closed_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed" markers.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for horizontal zeroline.
            vline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for vertical zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> price = pd.Series([1., 2., 3., 4., 5., 6., 5., 3., 2., 1.])
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> orders = pd.Series([1., -0.5, 0., -0.5, 2., 0., -0.5, -0.5, 0., -0.5])
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> trades = pf.trades
            >>> trades.plot_against_pnl("MFE")
            ```

            ![](/assets/images/api/trades_plot_against_pnl.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[xaxis]["tickformat"] = ".2%"
            def_layout_kwargs[xaxis]["title"] = "Return"
        else:
            def_layout_kwargs[xaxis]["title"] = "PnL"
        if field_pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
        def_layout_kwargs[yaxis]["title"] = field_label
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=returns[mask] if pct_scale else pnl[mask],
                            y=field[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=7,
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(neutral_mask, "Closed", plotting_cfg["contrast_color_schema"]["gray"], closed_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(open_mask, "Open", plotting_cfg["contrast_color_schema"]["orange"], open_trace_kwargs)

        # Plot zerolines
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref=xref,
                    yref="paper",
                    x0=0,
                    y0=y_domain[0],
                    x1=0,
                    y1=y_domain[1],
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                vline_shape_kwargs,
            )
        )
        return fig

    plot_mfe = partialmethod(
        plot_against_pnl,
        field="mfe",
        field_label="MFE",
        pct_scale=False,
    )
    """`Trades.plot_against_pnl` for `Trades.mfe`."""

    plot_mae = partialmethod(
        plot_against_pnl,
        field="mae",
        field_label="MAE",
        pct_scale=False,
    )
    """`Trades.plot_against_pnl` for `Trades.mae`."""

    plot_mfe_returns = partialmethod(
        plot_against_pnl,
        field="mfe_returns",
        field_label="MFE Return",
        pct_scale=True,
        field_pct_scale=True,
    )
    """`Trades.plot_against_pnl` for `Trades.mfe_returns`."""

    plot_mae_returns = partialmethod(
        plot_against_pnl,
        field="mae_returns",
        field_label="MAE Return",
        pct_scale=True,
        field_pct_scale=True,
    )
    """`Trades.plot_against_pnl` for `Trades.mae_returns`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        exit_profit_trace_kwargs: tp.KwargsLike = None,
        exit_loss_trace_kwargs: tp.KwargsLike = None,
        active_trace_kwargs: tp.KwargsLike = None,
        profit_shape_kwargs: tp.KwargsLike = None,
        loss_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            plot_markers (bool): Whether to plot markers.
            plot_zones (bool): Whether to plot zones.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Trades.close`.
            entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Entry" markers.
            exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit" markers.
            exit_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Profit" markers.
            exit_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Loss" markers.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Active" markers.
            profit_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for profit zones.
            loss_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for loss zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> import vectorbtpro as vbt

            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], name='Price')
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot()
            ```

            ![](/assets/images/api/trades_plot.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if entry_trace_kwargs is None:
            entry_trace_kwargs = {}
        if exit_trace_kwargs is None:
            exit_trace_kwargs = {}
        if exit_profit_trace_kwargs is None:
            exit_profit_trace_kwargs = {}
        if exit_loss_trace_kwargs is None:
            exit_loss_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if profit_shape_kwargs is None:
            profit_shape_kwargs = {}
        if loss_shape_kwargs is None:
            loss_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            entry_idx = self_col.get_map_field_to_index("entry_idx", minus_one_to_zero=True)
            entry_price = self_col.get_field_arr("entry_price")
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            exit_price = self_col.get_field_arr("exit_price")
            pnl = self_col.get_field_arr("pnl")
            status = self_col.get_field_arr("status")

            duration = to_1d_array(self_col.wrapper.to_timedelta(
                self_col.duration.values,
                to_pd=True,
                silence_warnings=True
            ).astype(str))

            if plot_markers:
                # Plot Entry markers
                if self_col.get_field_setting("parent_id", "ignore", False):
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id", "entry_idx", "size", "entry_price", "entry_fees", "direction"]
                    )
                else:
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id", "parent_id", "entry_idx", "size", "entry_price", "entry_fees", "direction"]
                    )
                _entry_trace_kwargs = merge_dicts(
                    dict(
                        x=entry_idx,
                        y=entry_price,
                        mode="markers",
                        marker=dict(
                            symbol="square",
                            color=plotting_cfg["contrast_color_schema"]["blue"],
                            size=7,
                            line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["blue"])),
                        ),
                        name="Entry",
                        customdata=entry_customdata,
                        hovertemplate=entry_hovertemplate,
                    ),
                    entry_trace_kwargs,
                )
                entry_scatter = go.Scatter(**_entry_trace_kwargs)
                fig.add_trace(entry_scatter, **add_trace_kwargs)

                # Plot end markers
                def _plot_end_markers(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                    if np.any(mask):
                        if self_col.get_field_setting("parent_id", "ignore", False):
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        else:
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "parent_id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        _kwargs = merge_dicts(
                            dict(
                                x=exit_idx[mask],
                                y=exit_price[mask],
                                mode="markers",
                                marker=dict(
                                    symbol="square",
                                    color=color,
                                    size=7,
                                    line=dict(width=1, color=adjust_lightness(color)),
                                ),
                                name=name,
                                customdata=exit_customdata,
                                hovertemplate=exit_hovertemplate,
                            ),
                            kwargs,
                        )
                        scatter = go.Scatter(**_kwargs)
                        fig.add_trace(scatter, **add_trace_kwargs)

                # Plot Exit markers
                _plot_end_markers(
                    (status == TradeStatus.Closed) & (pnl == 0.0),
                    "Exit",
                    plotting_cfg["contrast_color_schema"]["gray"],
                    exit_trace_kwargs,
                )

                # Plot Exit - Profit markers
                _plot_end_markers(
                    (status == TradeStatus.Closed) & (pnl > 0.0),
                    "Exit - Profit",
                    plotting_cfg["contrast_color_schema"]["green"],
                    exit_profit_trace_kwargs,
                )

                # Plot Exit - Loss markers
                _plot_end_markers(
                    (status == TradeStatus.Closed) & (pnl < 0.0),
                    "Exit - Loss",
                    plotting_cfg["contrast_color_schema"]["red"],
                    exit_loss_trace_kwargs,
                )

                # Plot Active markers
                _plot_end_markers(
                    status == TradeStatus.Open,
                    "Active",
                    plotting_cfg["contrast_color_schema"]["orange"],
                    active_trace_kwargs,
                )

            if plot_zones:
                profit_mask = pnl > 0.0
                if np.any(profit_mask):
                    # Plot profit zones
                    for i in np.flatnonzero(profit_mask):
                        fig.add_shape(
                            **merge_dicts(
                                dict(
                                    type="rect",
                                    xref=xref,
                                    yref=yref,
                                    x0=entry_idx[i],
                                    y0=entry_price[i],
                                    x1=exit_idx[i],
                                    y1=exit_price[i],
                                    fillcolor="green",
                                    opacity=0.2,
                                    layer="below",
                                    line_width=0,
                                ),
                                profit_shape_kwargs,
                            )
                        )

                loss_mask = pnl < 0.0
                if np.any(loss_mask):
                    # Plot loss zones
                    for i in np.flatnonzero(loss_mask):
                        fig.add_shape(
                            **merge_dicts(
                                dict(
                                    type="rect",
                                    xref=xref,
                                    yref=yref,
                                    x0=entry_idx[i],
                                    y0=entry_price[i],
                                    x1=exit_idx[i],
                                    y1=exit_price[i],
                                    fillcolor="red",
                                    opacity=0.2,
                                    layer="below",
                                    line_width=0,
                                ),
                                loss_shape_kwargs,
                            )
                        )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.plots`.

        Merges `vectorbtpro.generic.ranges.Ranges.plots_defaults` and
        `plots` from `vectorbtpro._settings.trades`."""
        from vectorbtpro._settings import settings

        trades_plots_cfg = settings["trades"]["plots"]

        return merge_dicts(Ranges.plots_defaults.__get__(self), trades_plots_cfg)

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot",
                tags="trades",
            ),
            plot_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func="plot_pnl",
                tags="trades",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Trades.override_field_config_doc(__pdoc__)
Trades.override_metrics_doc(__pdoc__)
Trades.override_subplots_doc(__pdoc__)

# ############# EntryTrades ############# #

entry_trades_field_config = ReadonlyConfig(
    dict(settings={"id": dict(title="Entry Trade Id"), "idx": dict(name="entry_idx")})  # remap field of Records,
)
"""_"""

__pdoc__[
    "entry_trades_field_config"
] = f"""Field config for `EntryTrades`.

```python
{entry_trades_field_config.prettify()}
```
"""

EntryTradesT = tp.TypeVar("EntryTradesT", bound="EntryTrades")


@override_field_config(entry_trades_field_config)
class EntryTrades(Trades):
    """Extends `Trades` for working with entry trade records."""

    @classmethod
    def from_orders(
        cls: tp.Type[EntryTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> EntryTradesT:
        """Build `EntryTrades` from `vectorbtpro.portfolio.orders.Orders`."""
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_entry_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )


# ############# ExitTrades ############# #

exit_trades_field_config = ReadonlyConfig(dict(settings={"id": dict(title="Exit Trade Id")}))
"""_"""

__pdoc__[
    "exit_trades_field_config"
] = f"""Field config for `ExitTrades`.

```python
{exit_trades_field_config.prettify()}
```
"""

ExitTradesT = tp.TypeVar("ExitTradesT", bound="ExitTrades")


@override_field_config(exit_trades_field_config)
class ExitTrades(Trades):
    """Extends `Trades` for working with exit trade records."""

    @classmethod
    def from_orders(
        cls: tp.Type[ExitTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> ExitTradesT:
        """Build `ExitTrades` from `vectorbtpro.portfolio.orders.Orders`."""
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_exit_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )


# ############# Positions ############# #

positions_field_config = ReadonlyConfig(
    dict(settings={"id": dict(title="Position Id"), "parent_id": dict(title="Parent Id", ignore=True)}),
)
"""_"""

__pdoc__[
    "positions_field_config"
] = f"""Field config for `Positions`.

```python
{positions_field_config.prettify()}
```
"""

PositionsT = tp.TypeVar("PositionsT", bound="Positions")


@override_field_config(positions_field_config)
class Positions(Trades):
    """Extends `Trades` for working with position records."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_trades(
        cls: tp.Type[PositionsT],
        trades: Trades,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PositionsT:
        """Build `Positions` from `Trades`."""
        if open is None:
            open = trades._open
        if high is None:
            high = trades._high
        if low is None:
            low = trades._low
        if close is None:
            close = trades._close
        func = jit_reg.resolve_option(nb.get_positions_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        position_records_arr = func(trades.values, trades.col_mapper.col_map)
        return cls.from_records(
            trades.wrapper,
            position_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )
