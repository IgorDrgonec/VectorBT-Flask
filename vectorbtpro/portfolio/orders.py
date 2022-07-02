# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with order records.

Order records capture information on filled orders. Orders are mainly populated when simulating
a portfolio and can be accessed as `vectorbtpro.portfolio.base.Portfolio.orders`.

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbtpro as vbt

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

>>> pf.orders.side_buy.count()
symbol
a    17
b    15
Name: count, dtype: int64

>>> pf.orders.side_sell.count()
symbol
a    24
b    26
Name: count, dtype: int64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Orders.metrics`.

```pycon
>>> pf.orders['a'].stats()
Start                  2019-12-31 22:00:00+00:00
End                    2020-02-29 22:00:00+00:00
Period                          61 days 00:00:00
Total Records                                 41
Side Counts: Buy                              17
Side Counts: Sell                             24
Size: Min              0 days 19:33:05.006182372
Size: Median                     1 days 00:00:00
Size: Max                        1 days 00:00:00
Fees: Min              0 days 20:26:25.905776572
Fees: Median           0 days 22:46:22.693324744
Fees: Max              1 days 01:04:25.541681491
Weighted Buy Price                      94.69917
Weighted Sell Price                    95.742148
Name: a, dtype: object
```

`Orders.stats` also supports (re-)grouping:

```pycon
>>> pf.orders.stats(group_by=True)
Start                  2019-12-31 22:00:00+00:00
End                    2020-02-29 22:00:00+00:00
Period                          61 days 00:00:00
Total Records                                 82
Side Counts: Buy                              32
Side Counts: Sell                             50
Size: Min              0 days 19:33:05.006182372
Size: Median                     1 days 00:00:00
Size: Max                        1 days 00:00:00
Fees: Min              0 days 20:26:25.905776572
Fees: Median           0 days 23:58:29.773897679
Fees: Max              1 days 02:29:08.904770159
Weighted Buy Price                     98.804452
Weighted Sell Price                    99.969934
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Orders.subplots`.

`Orders` class has a single subplot based on `Orders.plot`:

```pycon
>>> pf.orders['a'].plots()
```

![](/assets/images/orders_plots.svg)
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_dict
from vectorbtpro.generic.price_records import PriceRecords
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import order_dt, OrderSide, fs_order_dt, OrderType
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.records.decorators import attach_fields, override_field_config, attach_shortcut_properties
from vectorbtpro.signals.enums import StopType
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.template import Sub

__pdoc__ = {}

orders_field_config = ReadonlyConfig(
    dict(
        dtype=order_dt,
        settings=dict(
            id=dict(title="Order Id"),
            size=dict(title="Size"),
            price=dict(title="Price"),
            fees=dict(title="Fees"),
            side=dict(title="Side", mapping=OrderSide),
        ),
    )
)
"""_"""

__pdoc__[
    "orders_field_config"
] = f"""Field config for `Orders`.

```python
{orders_field_config.prettify()}
```
"""

orders_attach_field_config = ReadonlyConfig(dict(side=dict(attach_filters=True)))
"""_"""

__pdoc__[
    "orders_attach_field_config"
] = f"""Config of fields to be attached to `Orders`.

```python
{orders_attach_field_config.prettify()}
```
"""

orders_shortcut_config = ReadonlyConfig(
    dict(
        weighted_price=dict(obj_type="red_array"),
    )
)
"""_"""

__pdoc__[
    "orders_shortcut_config"
] = f"""Config of shortcut properties to be attached to `Orders`.

```python
{orders_shortcut_config.prettify()}
```
"""

OrdersT = tp.TypeVar("OrdersT", bound="Orders")


@attach_shortcut_properties(orders_shortcut_config)
@attach_fields(orders_attach_field_config)
@override_field_config(orders_field_config)
class Orders(PriceRecords):
    """Extends `vectorbtpro.generic.price_records.PriceRecords` for working with order records."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    # ############# Stats ############# #

    def get_weighted_price(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get size-weighted price average."""
        wrap_kwargs = merge_dicts(dict(name_or_index="weighted_price"), wrap_kwargs)
        return MappedArray.reduce(
            nb.weighted_price_reduce_meta_nb,
            self.get_field_arr("size"),
            self.get_field_arr("price"),
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            col_mapper=self.col_mapper,
            **kwargs,
        )

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Orders.stats`.

        Merges `vectorbtpro.generic.price_records.PriceRecords.stats_defaults` and
        `stats` from `vectorbtpro._settings.orders`."""
        from vectorbtpro._settings import settings

        orders_stats_cfg = settings["orders"]["stats"]

        return merge_dicts(PriceRecords.stats_defaults.__get__(self), orders_stats_cfg)

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
            side_counts=dict(
                title="Side Counts",
                calc_func="side.value_counts",
                incl_all_keys=True,
                post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
                tags=["orders", "side"],
            ),
            size=dict(
                title="Size",
                calc_func="size.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                apply_to_timedelta=True,
                tags=["orders", "size"],
            ),
            fees=dict(
                title="Fees",
                calc_func="fees.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                apply_to_timedelta=True,
                tags=["orders", "fees"],
            ),
            weighted_buy_price=dict(
                title="Weighted Buy Price",
                calc_func="side_buy.get_weighted_price",
                tags=["orders", "buy", "price"],
            ),
            weighted_sell_price=dict(
                title="Weighted Sell Price",
                calc_func="side_sell.get_weighted_price",
                tags=["orders", "sell", "price"],
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
        plot_ohlc: bool = True,
        plot_close: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        buy_trace_kwargs: tp.KwargsLike = None,
        sell_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Orders.close`.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        To display additional field on hover, define `customdata_index` as a key for a field in the field config
        and specify the index where to insert the data as a value. Additionally, you can define `hover_template`
        such as using `vectorbtpro.utils.template.Sub` where `title` is substituted by the title and
        `index` is substituted by (final) index in the customdata. If provided as a string, will be wrapped
        with `vectorbtpro.utils.template.Sub`. Defaults to "$title: %{{customdata[$index]}}". Enable
        `customdata_as_str` to stringify the data.

        Usage:
            ```pycon
            >>> price = pd.Series([1., 2., 3., 2., 1.], name='Price')
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> size = pd.Series([1., 1., 1., 1., -1.])
            >>> orders = vbt.Portfolio.from_orders(price, size).orders

            >>> orders.plot()
            ```

            ![](/assets/images/orders_plot.svg)
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
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot price
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
                ohlc_trace_kwargs["opacity"] = 0.65
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.plot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            id_ = self_col.get_field_arr("id")
            id_title = self_col.get_field_title("id")
            id_hover_template = self_col.get_field_setting(
                "id",
                "hover_template",
                "$title: %{customdata[$index]}",
            )
            if isinstance(id_hover_template, str):
                id_hover_template = Sub(id_hover_template)

            idx = self_col.get_map_field_to_index("idx")
            idx_title = self_col.get_field_title("idx")
            idx_hover_template = self_col.get_field_setting(
                "idx",
                "hover_template",
                "$title: %{customdata[$index]}",
            )
            if isinstance(idx_hover_template, str):
                idx_hover_template = Sub(idx_hover_template)

            size = self_col.get_field_arr("size")
            size_title = self_col.get_field_title("size")
            size_hover_template = self_col.get_field_setting(
                "size",
                "hover_template",
                "$title: %{customdata[$index]:,}",
            )
            if isinstance(size_hover_template, str):
                size_hover_template = Sub(size_hover_template)

            fees = self_col.get_field_arr("fees")
            fees_title = self_col.get_field_title("fees")
            fees_hover_template = self_col.get_field_setting(
                "fees",
                "hover_template",
                "$title: %{customdata[$index]:,}",
            )
            if isinstance(fees_hover_template, str):
                fees_hover_template = Sub(fees_hover_template)

            price = self_col.get_field_arr("price")
            price_title = self_col.get_field_title("price")
            price_hover_template = self_col.get_field_setting(
                "price",
                "hover_template",
                "$title: %{customdata[$index]:,}",
            )
            if isinstance(price_hover_template, str):
                price_hover_template = Sub(price_hover_template)

            side = self_col.get_field_arr("side")

            def _prepare_customdata(mask):
                customdata_info = [
                    (id_, id_title, id_hover_template),
                    (idx.astype(str).values, idx_title, idx_hover_template),
                    (price, price_title, price_hover_template),
                    (size, size_title, size_hover_template),
                    (fees, fees_title, fees_hover_template),
                ]
                dtype = self.field_config.get("dtype")
                for field in dtype.names:
                    field_customdata_index = self_col.get_field_setting(field, "customdata_index", None)
                    if field_customdata_index is not None:
                        if not isinstance(field_customdata_index, int):
                            raise ValueError("Setting customdata_index must be an integer or None")
                        if field_customdata_index < 0:
                            field_customdata_index = len(customdata_info) + field_customdata_index + 1
                        field_hover_template = self_col.get_field_setting(
                            field,
                            "hover_template",
                            "$title: %{customdata[$index]}",
                        )
                        if isinstance(field_hover_template, str):
                            field_hover_template = Sub(field_hover_template)
                        field_title = self_col.get_field_title(field)
                        customdata_as_str = self_col.get_field_setting(field, "customdata_as_str", False)
                        if customdata_as_str:
                            field_arr = self_col.get_apply_mapping_str_arr(field)
                        else:
                            field_arr = self_col.get_apply_mapping_arr(field)
                        customdata_info.insert(field_customdata_index, (field_arr, field_title, field_hover_template))
                customdata = []
                hovertemplate = []
                for i in range(len(customdata_info)):
                    customdata.append(customdata_info[i][0][mask])
                    _hovertemplate = customdata_info[i][2].substitute(dict(title=customdata_info[i][1], index=i))
                    if not _hovertemplate.startswith("<br>"):
                        _hovertemplate = "<br>" + _hovertemplate
                    hovertemplate.append(_hovertemplate)
                return np.stack(customdata, axis=1), "\n".join(hovertemplate)

            buy_mask = side == OrderSide.Buy
            if buy_mask.any():
                # Plot buy markers
                buy_customdata, buy_hovertemplate = _prepare_customdata(buy_mask)
                buy_scatter = go.Scatter(
                    x=idx[buy_mask],
                    y=price[buy_mask],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        color=plotting_cfg["contrast_color_schema"]["green"],
                        size=8,
                        line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["green"])),
                    ),
                    name="Buy",
                    customdata=buy_customdata,
                    hovertemplate=buy_hovertemplate,
                )
                buy_scatter.update(**buy_trace_kwargs)
                fig.add_trace(buy_scatter, **add_trace_kwargs)

            sell_mask = side == OrderSide.Sell
            if sell_mask.any():
                # Plot sell markers
                sell_customdata, sell_hovertemplate = _prepare_customdata(sell_mask)
                sell_scatter = go.Scatter(
                    x=idx[sell_mask],
                    y=price[sell_mask],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        color=plotting_cfg["contrast_color_schema"]["red"],
                        size=8,
                        line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["red"])),
                    ),
                    name="Sell",
                    customdata=sell_customdata,
                    hovertemplate=sell_hovertemplate,
                )
                sell_scatter.update(**sell_trace_kwargs)
                fig.add_trace(sell_scatter, **add_trace_kwargs)

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Orders.plots`.

        Merges `vectorbtpro.generic.price_records.PriceRecords.plots_defaults` and
        `plots` from `vectorbtpro._settings.orders`."""
        from vectorbtpro._settings import settings

        orders_plots_cfg = settings["orders"]["plots"]

        return merge_dicts(PriceRecords.plots_defaults.__get__(self), orders_plots_cfg)

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=dict(
                title="Orders",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot",
                tags="orders",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Orders.override_field_config_doc(__pdoc__)
Orders.override_metrics_doc(__pdoc__)
Orders.override_subplots_doc(__pdoc__)


fs_orders_field_config = ReadonlyConfig(
    dict(
        dtype=fs_order_dt,
        settings=dict(
            idx=dict(title="Fill Timestamp"),
            signal_idx=dict(
                title="Signal Timestamp",
                mapping="index",
                noindex=True,
                customdata_as_str=True,
                customdata_index=1,
            ),
            creation_idx=dict(
                title="Creation Timestamp",
                mapping="index",
                noindex=True,
                customdata_as_str=True,
                customdata_index=2,
            ),
            type=dict(
                title="Type",
                mapping=OrderType,
                customdata_index=-1,
                customdata_as_str=True,
            ),
            stop_type=dict(
                title="Stop Type",
                mapping=StopType,
                customdata_index=-1,
                customdata_as_str=True,
            ),
        ),
    )
)
"""_"""

__pdoc__[
    "fs_orders_field_config"
] = f"""Field config for `FSOrders`.

```python
{fs_orders_field_config.prettify()}
```
"""

fs_orders_attach_field_config = ReadonlyConfig(
    dict(
        type=dict(attach_filters=True),
        stop_type=dict(attach_filters=True),
    )
)
"""_"""

__pdoc__[
    "fs_orders_attach_field_config"
] = f"""Config of fields to be attached to `FSOrders`.

```python
{fs_orders_attach_field_config.prettify()}
```
"""

fs_orders_shortcut_config = ReadonlyConfig(
    dict(
        signal_to_creation_duration=dict(obj_type="mapped_array"),
        creation_to_fill_duration=dict(obj_type="mapped_array"),
        signal_to_fill_duration=dict(obj_type="mapped_array"),
    )
)
"""_"""

__pdoc__[
    "fs_orders_shortcut_config"
] = f"""Config of shortcut properties to be attached to `FSOrders`.

```python
{fs_orders_shortcut_config.prettify()}
```
"""

FSOrdersT = tp.TypeVar("FSOrdersT", bound="FSOrders")


@attach_shortcut_properties(fs_orders_shortcut_config)
@attach_fields(fs_orders_attach_field_config)
@override_field_config(fs_orders_field_config)
class FSOrders(Orders):
    """Extends `Orders` for working with order records generated from signals."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    def get_signal_to_creation_duration(self, **kwargs) -> MappedArray:
        """Get duration between signal and creation."""
        duration = self.get_field_arr("creation_idx") - self.get_field_arr("signal_idx")
        return self.map_array(duration, **kwargs)

    def get_creation_to_fill_duration(self, **kwargs) -> MappedArray:
        """Get duration between creation and fill."""
        duration = self.get_field_arr("idx") - self.get_field_arr("creation_idx")
        return self.map_array(duration, **kwargs)

    def get_signal_to_fill_duration(self, **kwargs) -> MappedArray:
        """Get duration between signal and fill."""
        duration = self.get_field_arr("idx") - self.get_field_arr("signal_idx")
        return self.map_array(duration, **kwargs)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        start=Orders.metrics["start"],
        end=Orders.metrics["end"],
        period=Orders.metrics["period"],
        total_records=Orders.metrics["total_records"],
        side_counts=Orders.metrics["side_counts"],
        type_counts=dict(
            title="Type Counts",
            calc_func="type.value_counts",
            incl_all_keys=True,
            post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
            tags=["orders", "type"],
        ),
        stop_type_counts=dict(
            title="Stop Type Counts",
            calc_func="stop_type.value_counts",
            incl_all_keys=True,
            post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
            tags=["orders", "stop_type"],
        ),
        size=Orders.metrics["size"],
        fees=Orders.metrics["fees"],
        weighted_buy_price=Orders.metrics["weighted_buy_price"],
        weighted_sell_price=Orders.metrics["weighted_sell_price"],
        avg_signal_to_creation_duration=dict(
            title="Avg Signal-Creation Duration",
            calc_func="signal_to_creation_duration.mean",
            apply_to_timedelta=True,
            tags=["orders", "duration"],
        ),
        avg_creation_to_fill_duration=dict(
            title="Avg Creation-Fill Duration",
            calc_func="creation_to_fill_duration.mean",
            apply_to_timedelta=True,
            tags=["orders", "duration"],
        ),
        avg_signal_to_fill_duration=dict(
            title="Avg Signal-Fill Duration",
            calc_func="signal_to_fill_duration.mean",
            apply_to_timedelta=True,
            tags=["orders", "duration"],
        ),
    )

    @property
    def metrics(self) -> Config:
        return self._metrics


FSOrders.override_field_config_doc(__pdoc__)
FSOrders.override_metrics_doc(__pdoc__)
