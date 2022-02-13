# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Plotly Express pandas accessors.

!!! note
    Accessors do not utilize caching."""

from vectorbtpro.utils.opt_packages import assert_can_import

assert_can_import("plotly")

from inspect import getmembers, isfunction

import pandas as pd
import plotly.express as px

from vectorbtpro import _typing as tp
from vectorbtpro.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic.plotting import clean_labels
from vectorbtpro.accessors import register_vbt_accessor, register_df_vbt_accessor, register_sr_vbt_accessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.figure import make_figure


def attach_px_methods(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """Class decorator to attach Plotly Express methods."""

    for px_func_name, px_func in getmembers(px, isfunction):
        if checks.func_accepts_arg(px_func, "data_frame") or px_func_name == "imshow":

            def plot_func(
                self,
                *args,
                _px_func_name: str = px_func_name,
                _px_func: tp.Callable = px_func,
                **kwargs,
            ) -> tp.BaseFigure:
                from vectorbtpro._settings import settings

                layout_cfg = settings["plotting"]["layout"]

                layout_kwargs = dict(
                    template=kwargs.pop("template", layout_cfg["template"]),
                    width=kwargs.pop("width", layout_cfg["width"]),
                    height=kwargs.pop("height", layout_cfg["height"]),
                )
                # Fix category_orders
                if "color" in kwargs:
                    if isinstance(kwargs["color"], str):
                        if isinstance(self.obj, pd.DataFrame):
                            if kwargs["color"] in self.obj.columns:
                                category_orders = dict()
                                category_orders[kwargs["color"]] = sorted(self.obj[kwargs["color"]].unique())
                                kwargs = merge_dicts(dict(category_orders=category_orders), kwargs)

                # Fix Series name
                obj = self.obj.copy(deep=False)
                if isinstance(obj, pd.Series):
                    if obj.name is not None:
                        obj = obj.rename(str(obj.name))
                else:
                    obj.columns = clean_labels(obj.columns)
                obj.index = clean_labels(obj.index)

                if _px_func_name == "imshow":
                    return make_figure(
                        _px_func(to_2d_array(obj), *args, **layout_kwargs, **kwargs),
                        layout=layout_kwargs,
                    )
                return make_figure(_px_func(obj, *args, **layout_kwargs, **kwargs), layout=layout_kwargs)

            setattr(cls, px_func_name, plot_func)
    return cls


@register_vbt_accessor("px")
@attach_px_methods
class PXAccessor(BaseAccessor):
    """Accessor for running Plotly Express functions.

    Accessible via `pd.Series.vbt.px` and `pd.DataFrame.vbt.px`.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> import vectorbtpro as vbt

        >>> pd.Series([1, 2, 3]).vbt.px.bar()
        ```

        ![](/assets/images/px_bar.svg)
    """

    def __init__(self, obj: tp.SeriesFrame, **kwargs) -> None:
        BaseAccessor.__init__(self, obj, **kwargs)


@register_sr_vbt_accessor("px")
class PXSRAccessor(PXAccessor, BaseSRAccessor):
    """Accessor for running Plotly Express functions. For Series only.

    Accessible via `pd.Series.vbt.px`."""

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        BaseSRAccessor.__init__(self, obj, **kwargs)
        PXAccessor.__init__(self, obj, **kwargs)


@register_df_vbt_accessor("px")
class PXDFAccessor(PXAccessor, BaseDFAccessor):
    """Accessor for running Plotly Express functions. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.px`."""

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        BaseDFAccessor.__init__(self, obj, **kwargs)
        PXAccessor.__init__(self, obj, **kwargs)
